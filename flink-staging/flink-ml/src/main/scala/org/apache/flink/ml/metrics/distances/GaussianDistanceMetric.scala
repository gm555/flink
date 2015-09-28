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

package org.apache.flink.ml.metrics.distances

import org.apache.flink.ml.math.Vector

/** This class implements a Gaussian distance metric.
  */
class GaussianDistanceMetric(standardDeviation: Double) extends DistanceMetric{
  override def distance(a: Vector, b: Vector): Double = {
    checkValidArguments(a, b)
    val sqEucDist = (0 until a.size).map(i => math.pow(a(i) - b(i), 2)).sum
    math.exp(-sqEucDist / (2 * standardDeviation * standardDeviation))  }
}

object GaussianDistanceMetric {
  def apply(standardDeviation: Double) = new GaussianDistanceMetric(standardDeviation)
}

