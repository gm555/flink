/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package org.apache.flink.runtime.operators.hash;

import java.io.IOException;
import java.util.List;

import org.apache.flink.api.common.functions.FlatJoinFunction;
import org.apache.flink.api.common.typeutils.TypeComparator;
import org.apache.flink.api.common.typeutils.TypePairComparator;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.core.memory.MemorySegment;
import org.apache.flink.runtime.io.disk.iomanager.IOManager;
import org.apache.flink.runtime.jobgraph.tasks.AbstractInvokable;
import org.apache.flink.runtime.memory.MemoryAllocationException;
import org.apache.flink.runtime.memory.MemoryManager;
import org.apache.flink.runtime.operators.util.JoinTaskIterator;
import org.apache.flink.util.Collector;
import org.apache.flink.util.MutableObjectIterator;


/**
 * An implementation of the {@link org.apache.flink.runtime.operators.util.JoinTaskIterator} that uses a hybrid-hash-join
 * internally to match the records with equal key. The build side of the hash is the second input of the match.  
 */
public class ReusingBuildSecondHashMatchIterator<V1, V2, O> extends HashMatchIteratorBase implements JoinTaskIterator<V1, V2, O> {
	
	protected final MutableHashTable<V2, V1> hashJoin;
	
	private final V2 nextBuildSideObject;
	
	private final V2 tempBuildSideRecord;
	
	protected final TypeSerializer<V1> probeSideSerializer;
	
	private final MemoryManager memManager;
	
	private final MutableObjectIterator<V1> firstInput;
	
	private final MutableObjectIterator<V2> secondInput;
	
	private volatile boolean running = true;
	
	// --------------------------------------------------------------------------------------------
	
	public ReusingBuildSecondHashMatchIterator(
			MutableObjectIterator<V1> firstInput,
			MutableObjectIterator<V2> secondInput,
			TypeSerializer<V1> serializer1,
			TypeComparator<V1> comparator1,
			TypeSerializer<V2> serializer2,
			TypeComparator<V2> comparator2,
			TypePairComparator<V1, V2> pairComparator,
			MemoryManager memManager,
			IOManager ioManager,
			AbstractInvokable ownerTask,
			double memoryFraction,
			boolean useBitmapFilters) throws MemoryAllocationException {
		
		this.memManager = memManager;
		this.firstInput = firstInput;
		this.secondInput = secondInput;
		this.probeSideSerializer = serializer1;
		
		this.nextBuildSideObject = serializer2.createInstance();
		this.tempBuildSideRecord = serializer2.createInstance();

		this.hashJoin = getHashJoin(serializer2, comparator2, serializer1, comparator1, pairComparator,
			memManager, ioManager, ownerTask, memoryFraction, useBitmapFilters);
	}
	
	// --------------------------------------------------------------------------------------------
	
	@Override
	public void open() throws IOException, MemoryAllocationException, InterruptedException {
		this.hashJoin.open(this.secondInput, this.firstInput);
	}

	@Override
	public void close() {
		// close the join
		this.hashJoin.close();
		
		// free the memory
		final List<MemorySegment> segments = this.hashJoin.getFreedMemory();
		this.memManager.release(segments);
	}

	@Override
	public boolean callWithNextKey(FlatJoinFunction<V1, V2, O> matchFunction, Collector<O> collector)
	throws Exception
	{
		if (this.hashJoin.nextRecord())
		{
			// we have a next record, get the iterators to the probe and build side values
			final MutableHashTable.HashBucketIterator<V2, V1> buildSideIterator = this.hashJoin.getBuildSideIterator();
			V2 nextBuildSideRecord = this.nextBuildSideObject;
			
			// get the first build side value
			if ((nextBuildSideRecord = buildSideIterator.next(nextBuildSideRecord)) != null) {
				V2 tmpRec = this.tempBuildSideRecord;
				final V1 probeRecord = this.hashJoin.getCurrentProbeRecord();
				
				// check if there is another build-side value
				if ((tmpRec = buildSideIterator.next(tmpRec)) != null) {
					// call match on the first pair
					matchFunction.join(probeRecord, nextBuildSideRecord, collector);
					
					// call match on the second pair
					matchFunction.join(probeRecord, tmpRec, collector);
					
					while (this.running && ((nextBuildSideRecord = buildSideIterator.next(nextBuildSideRecord)) != null)) {
						// call match on the next pair
						// make sure we restore the value of the probe side record
						matchFunction.join(probeRecord, nextBuildSideRecord, collector);
					}
				}
				else {
					// only single pair matches
					matchFunction.join(probeRecord, nextBuildSideRecord, collector);
				}
			}
			return true;
		}
		else {
			return false;
		}
	}
	
	@Override
	public void abort() {
		this.running = false;
		this.hashJoin.abort();
	}
}
