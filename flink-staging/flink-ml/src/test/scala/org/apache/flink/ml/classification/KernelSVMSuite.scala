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

package org.apache.flink.ml.classification

import org.scalatest.{ FlatSpec, Matchers }
import org.apache.flink.ml.math.{ Vector => FlinkVector, DenseVector }
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.api.scala._
import org.apache.flink.test.util.FlinkTestBase
import org.apache.flink.ml.metrics.distances.GaussianDistanceMetric

class KernelSVMSuite extends FlatSpec with Matchers with FlinkTestBase {

  behavior of "The kernel SVM"

  it should "train a kernel SVM" in {

    val env = ExecutionEnvironment.getExecutionEnvironment

    val kernelSVM = KernelSVM()
      .setApproximationRatio(1.0)
      .setIterations(10)
      .setKernel(new GaussianDistanceMetric(0.2))
      .setLearningRate(0.1)
      .setRegularizationConstant(0.1)

    val trainingDS = env.fromCollection(NonLinearClassification.trainingData)

    kernelSVM.fit(trainingDS)

    val loss = kernelSVM.loss.collect().head

    loss should be(NonLinearClassification.expectedLoss +- 1e-9)
  }

  it should "make (mostly) correct predictions" in {

    val env = ExecutionEnvironment.getExecutionEnvironment

    val kernelSVM = KernelSVM()
      .setApproximationRatio(1.0)
      .setIterations(10)
      .setKernel(new GaussianDistanceMetric(0.2))
      .setLearningRate(0.1)
      .setRegularizationConstant(0.1)

    val trainingDS = env.fromCollection(NonLinearClassification.trainingData)

    kernelSVM.fit(trainingDS)

    val predictions = kernelSVM.predict(trainingDS)

    val absoluteErrorSum = predictions.collect().map {
      case (prediction: Double, LabeledVector(truth: Double, _)) =>
        math.abs(prediction - truth)
    }.sum

    absoluteErrorSum should be(0.0)

  }

}
