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

import org.apache.flink.ml.pipeline.{ Predictor, FitOperation, PredictDataSetOperation }
import org.apache.flink.ml.common._
import org.apache.flink.api.scala._
import scala.collection.mutable.ArrayBuffer
import org.apache.flink.ml.math._
import org.apache.flink.ml.optimization.LossFunction
import org.apache.flink.ml.optimization.SimpleGradientDescent
import org.apache.flink.ml.metrics.distances.DistanceMetric

/**
 * Implements a non linear kernel SVM using a reformulation of the Nyström approximation by
 * Dhruv Mahajan, S. Sathiya Keerthi and S. Sundararajan ("A Distributed Algorithm
 * for Training Nonlinear Kernel Machines", 2014).
 *
 * The kernel has to be positive definite.
 *
 * Labels take their values in {-1; 1}.
 * The basis points for the reduced kernel matrix are chosen randomly.
 * The minimization problem is solved using 'GradientDescent'.
 *
 * @example
 *          {{{
 *             // get training and test sets
 *             val trainingSet: DataSet[LabeledVector] = ...
 *             val testSet: DataSet[Vector] = ...
 *
 *             // scale features
 *             val scaler = StandardScaler()
 *             scaler.fit(trainingSet)
 *             val scaledTrainingSet = scaler.transform(trainingSet)
 *             val scaledTestSet = scaler.transform(testSet)
 *
 *             // get a positive definite kernel
 *             val gaussianKernel = GaussianDistanceMetric(0.2)
 *
 *             val kernelSVM = KernelSVM()
 *               .setApproximationRatio(0.1)
 *               .setIterations(10)
 *               .setKernel(gaussianKernel)
 *               .setLearningRate(0.1)
 *               .setRegularizationConstant(0.5)
 *
 *             kernelSVM.fit(scaledTrainingSet)
 *             val predictions: DataSet[LabeledVector] = svm.predict(scaledTestSet)
 *          }}}
 *
 * =Parameters=
 *
 *  - [[org.apache.flink.ml.classification.KernelSVM.approximationRatio]]:
 *  Sets the ratio to which the kernel matrix will be approximated. This
 *  number should be at most 1.0 (i.e. no approximation).
 *  (Default Value: '''0.1''')
 *  
 *  - [[org.apache.flink.ml.classification.KernelSVM.RegularizationConstant]]:
 *  Defines the regularization constant of the kernel SVM algorithm.
 *  (Default value: '''1.0''')
 *
 *  - [[org.apache.flink.ml.classification.KernelSVM.Iterations]]:
 *  Defines the number of iterations of the outer loop method.
 *  (Default value: '''10''')
 *
 *  - [[org.apache.flink.ml.classification.KernelSVM.LearningRate]]:
 *  Defines the initial step size for the updates of the weight vector. This value has to be tuned
 *  in case that the algorithm becomes instable. (Default value: '''0.1''')
 *
 *  - [[org.apache.flink.ml.classification.KernelSVM.Kernel]]:
 *  Defines the kernel to use. This kernel has to be positive definite.
 *  (Default value: '''None''')
 */
class KernelSVM extends Predictor[KernelSVM] {

  import KernelSVM._

  /** Information for the predict operation:
   *  (basis points, learned weight vector) */
  var learnedInfo: Option[DataSet[(WeightVector, Array[Vector])]] = None

  /** Information for monitoring */
  var loss: DataSet[Double] = null

  /**
   * Sets the ratio for the approximation
   * of the kernel matrix
   *
   * @param approximationRatio
   * @return itself
   */
  def setApproximationRatio(approximationRatio: Double): KernelSVM = {
    parameters.add(ApproximationRatio, approximationRatio)
    this
  }

  /**
   * Sets the regularization constant
   *
   * @param regularizationConstant
   * @return itself
   */
  def setRegularizationConstant(regularizationConstant: Double): KernelSVM = {
    parameters.add(RegularizationConstant, regularizationConstant)
    this
  }

  /**
   * Sets the number of iterations
   *
   * @param iterations
   * @return itself
   */
  def setIterations(iterations: Int): KernelSVM = {
    parameters.add(Iterations, iterations)
    this
  }

  /**
   * Sets the learning rate
   *
   * @param learningRate
   * @return itself
   */
  def setLearningRate(learningRate: Double): KernelSVM = {
    parameters.add(LearningRate, learningRate)
    this
  }

  /**
   * Sets the kernel
   *
   * @param kernel
   * @return itself
   */
  def setKernel(kernel: DistanceMetric): KernelSVM = {
    parameters.add(Kernel, kernel)
    this
  }

}

object KernelSVM {

  // ========================================== Parameters =========================================

  case object ApproximationRatio extends Parameter[Double] {
    val defaultValue = Some(0.1)
  }

  case object RegularizationConstant extends Parameter[Double] {
    val defaultValue = Some(0.)
  }

  case object Iterations extends Parameter[Int] {
    val defaultValue = Some(10)
  }

  case object LearningRate extends Parameter[Double] {
    val defaultValue = Some(0.1)
  }

  case object Kernel extends Parameter[DistanceMetric] {
    val defaultValue = None
  }

  // ========================================== Factory methods ====================================

  def apply(): KernelSVM = {
    new KernelSVM()
  }

  // ========================================== Operations =========================================

  /**
   * [[FitOperation]] which trains a kernel SVM based on a given training data set.
   *
   */
  implicit val fitKernelSVM = {
    new FitOperation[KernelSVM, LabeledVector] {
      override def fit(
        instance: KernelSVM,
        fitParameters: ParameterMap,
        input: DataSet[LabeledVector]): Unit = {
        val p = instance.parameters ++ fitParameters

        val approximationRatio = p(ApproximationRatio)
        val iterations = p(Iterations)
        val regularizationConstant = p(RegularizationConstant)
        val learningRate = p(LearningRate)
        val kernel = p(Kernel)

        val basisPoints = getBasisPoints(input, approximationRatio)
        val basisPointsArray = getBasisPointsArray(basisPoints)
        val kernelizedData = kernelizeTrainingSet(input, basisPointsArray, kernel)
        val reducedKernelMatrix = getKernelMatrix(basisPointsArray, kernel)

        val lossFunction = new KernelLossFunction()
        val gradientDescent = SimpleGradientDescent()

        val weights = kernelizedData.first(1).map(e =>
          WeightVector(DenseVector.init(e.vector.size, 0), 0))

        val newWeights = gradientDescent
          .setLossFunction(lossFunction)
          .setIterations(iterations)
          .setStepsize(learningRate)
          .setRegularizationConstant(regularizationConstant)
          .optimize(kernelizedData, Some(weights),
            reducedKernelMatrix)

        // Store the learned information in the given instance
        val info = newWeights.cross(basisPointsArray) {
          (l, r) => (l, r)
        }
        instance.learnedInfo = Some(info)

        // set the loss for this model in case we want to
        // try out different settings
        instance.loss = kernelizedData.mapWithBcVariable(newWeights) {
          (point, weights) =>
            (lossFunction.lossGradient(point, weights)._1, 1)
        }.reduce {
          (e1, e2) => ((e1._1 + e2._1), (e1._2 + e2._2))
        }.map(e => e._1 / e._2)
      }
    }
  }

  /**
   * [[org.apache.flink.ml.pipeline.PredictOperation]] for [[Vector]] types. The result type
   * is a [[LabeledVector]], corresponding to the point represented by the vector together
   * with the corresponding predicted label.
   *
   * @return A DataSet[LabeledVector].
   */
  implicit def predictLabels = {
    new PredictDataSetOperation[KernelSVM, DenseVector, LabeledVector] {
      override def predictDataSet(
        instance: KernelSVM,
        predictParameters: ParameterMap,
        input: DataSet[DenseVector]): DataSet[LabeledVector] = {
        val p = instance.parameters ++ predictParameters

        val kernel = p(Kernel)

        instance.learnedInfo match {
          case Some(info) => {
            input.mapWithBcVariable(info) {
              (point, info) =>
                val (weights, basisPoints) = info
                val kernelizedPoint = kernelizePoint(point,
                  basisPoints, kernel)
                val dot = BLAS.dot(kernelizedPoint, weights.weights)
                val label = if (dot > 0) 1 else -1
                LabeledVector(label, point)
            }
          }
          case None => {
            throw new RuntimeException("The KernelSVM model has not been trained. Call first fit" +
              "before calling the predict operation.")
          }
        }
      }
    }
  }
  
    /**
   * [[org.apache.flink.ml.pipeline.PredictOperation]] for [[LabeledVector]] types. The result type
   * is a [[(Int, LabeledVector)]], with the Double corresponding to the predicted label.
   * The LabeledVector is not modified.
   * Convenient format for testing different models.
   *
   * @return A DataSet[LabeledVector].
   */
  implicit def predictLabelsForComparison = {
    new PredictDataSetOperation[KernelSVM, LabeledVector, (Double, LabeledVector)] {
      override def predictDataSet(
        instance: KernelSVM,
        predictParameters: ParameterMap,
        input: DataSet[LabeledVector]): DataSet[(Double, LabeledVector)] = {
        val p = instance.parameters ++ predictParameters

        val kernel = p(Kernel)

        instance.learnedInfo match {
          case Some(info) => {
            input.mapWithBcVariable(info) {
              (point, info) =>
                val (weights, basisPoints) = info
                val kernelizedPoint = kernelizePoint(point.vector,
                  basisPoints, kernel)
                val dot = BLAS.dot(kernelizedPoint, weights.weights)
                val label = if (dot > 0) 1 else -1
                (label, point)
            }
          }
          case None => {
            throw new RuntimeException("The KernelSVM model has not been trained. Call first fit" +
              "before calling the predict operation.")
          }
        }
      }
    }
  }

  // ========================================== Auxiliary functions ================================

  def getBasisPoints(data: DataSet[LabeledVector],
    approximationRatio: Double): DataSet[Vector] = {
    data.filter(e => scala.util.Random.nextDouble < approximationRatio)
      .map(e => e.vector)
  }

  def getBasisPointsArray(
    basisPoints: DataSet[Vector]): DataSet[Array[Vector]] = {
    basisPoints
      .reduceGroup { point =>
        val res = new ArrayBuffer[Vector]
        while (point.hasNext) res += point.next
        res.toArray
      }
  }

  def kernelizeTrainingSet(data: DataSet[LabeledVector],
    basisPointsArray: DataSet[Array[Vector]],
    kernel: DistanceMetric): DataSet[LabeledVector] = {
    data.mapWithBcVariable(basisPointsArray) {
      (trainingExample, basisPoints) =>
        LabeledVector(trainingExample.label,
          kernelizePoint(trainingExample.vector,
            basisPoints, kernel))
    }
  }

  def kernelizePoint(point: Vector,
    basisPointsArray: Array[Vector],
    kernel: DistanceMetric): Vector = {
    val transition = new Array[Double](basisPointsArray.length)

    for (i <- 0 until transition.length) {
      transition(i) = kernel.distance(point, basisPointsArray(i))
    }
    new DenseVector(transition)
  }

  // since the reduced kernel matrix is symmetric,
  // we store only the upper triangular matrix
  def getKernelMatrix(pointsArray: DataSet[Array[Vector]],
    kernel: DistanceMetric): DataSet[Array[Double]] = {
    pointsArray.map { p =>
      val order = p.length
      val matrix = new Array[Double](order * (order + 1) / 2)
      var k = 0
      for (i <- 0 until order) {
        for (j <- 0 to i) {
          matrix(k) = kernel.distance(p(i), p(j))
          k = k + 1
        }
      }
      matrix
    }
  }

}

case class KernelLossFunction extends LossFunction {

  def lossGradient(dataPoint: LabeledVector,
    weightVector: WeightVector): (Double, WeightVector) = {

    val dot = BLAS.dot(dataPoint.vector, weightVector.weights) +
      weightVector.intercept

    val marginEncroachmentCosts = 1 - dot * dataPoint.label
    val marginEncroachments =
      if (marginEncroachmentCosts > 0) 1 else 0

    val loss = 0.5 * marginEncroachments *
      marginEncroachmentCosts * marginEncroachmentCosts

    val scaler = marginEncroachments * (dot - dataPoint.label)
    BLAS.scal(scaler, dataPoint.vector)

    (loss, WeightVector(dataPoint.vector, scaler))
  }
}
