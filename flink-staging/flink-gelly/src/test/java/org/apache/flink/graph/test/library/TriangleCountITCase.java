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

package org.apache.flink.graph.test.library;

import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple1;
import org.apache.flink.graph.Graph;
import org.apache.flink.graph.example.utils.TriangleCountData;
import org.apache.flink.graph.library.GSATriangleCount;
import org.apache.flink.test.util.MultipleProgramsTestBase;
import org.apache.flink.types.NullValue;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.List;

@RunWith(Parameterized.class)
public class TriangleCountITCase extends MultipleProgramsTestBase {

	private String expectedResult;

	public TriangleCountITCase(TestExecutionMode mode) {
		super(mode);
	}

	@Test
	public void testGSATriangleCount() throws Exception {

		ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

		Graph<Long, NullValue, NullValue> graph = Graph.fromDataSet(TriangleCountData.getDefaultEdgeDataSet(env),
				env).getUndirected();

		List<Tuple1<Integer>> numberOfTriangles = graph.run(new GSATriangleCount()).collect();
		expectedResult = TriangleCountData.RESULTED_NUMBER_OF_TRIANGLES;

		compareResultAsTuples(numberOfTriangles, expectedResult);
	}
}
