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
 * The minimization problem is solved by applying batch gradient descent.
 *
 * @example
 *          {{{
 *             // get training and test sets
 *             val trainingSet: DataSet[LabeledVector] = ...
 *             val testSet: DataSet[Vector] = ...
 *
 *             // get a positive definite kernel
 *             val gaussianKernel = new GaussianDistanceMetric(0.2)
 *
 *             val kernelSVM = KernelSVM()
 *               .setApproximationRatio(0.1)
 *               .setIterations(10)
 *               .setKernel(gaussianKernel)
 *               .setLearningRate(0.1)
 *               .setRegularizationConstant(0.5)
 *
 *             kernelSVM.fit(trainingSet)
 *             val predictions: DataSet[LabeledVector] = svm.predict(testSet)
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

  /**
   * Information for the predict operation:
   *  (basis points, learned weight vector)
   */
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
    val defaultValue = Some(1.0)
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

        val markedData = selectBasisPoints(input, approximationRatio)
        val basisPointsArray = getBasisPointsArray(markedData)
        val kernelizedData = kernelizeTrainingSet(markedData, basisPointsArray, kernel)

        val weights = kernelizedData.first(1).map(e =>
          WeightVector(DenseVector.init(e._2.vector.size, 0), 0))

        val newWeights = weights.iterate(iterations) {
          weightVectorDS =>
            batchGradientDescentStep(kernelizedData, weightVectorDS,
              regularizationConstant, learningRate)
        }

        // Stores the learned information in the given instance
        val info = newWeights.cross(basisPointsArray) {
          (l, r) => (l, r)
        }
        instance.learnedInfo = Some(info)

        // sets the loss for this model in case we want to
        // try out different settings
        instance.loss = kernelizedData.mapWithBcVariable(newWeights) {
          (point, weights) =>
            (getUnregularizedLoss(point._2, weights), 1)
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

  private def selectBasisPoints(data: DataSet[LabeledVector],
    approximationRatio: Double): DataSet[(Boolean, LabeledVector)] = {
    data.map(e => (scala.util.Random.nextDouble < approximationRatio, e))
  }

  private def getBasisPointsArray(
    markedData: DataSet[(Boolean, LabeledVector)]): DataSet[Array[Vector]] = {
    markedData.map(e => (e._1, ArrayBuffer(e._2.vector)))
      .reduce {
        (a, b) =>
          val buffer = new ArrayBuffer[Vector]
          if (a._1) {
            buffer.++=(a._2)
          }
          if (b._1) {
            buffer.++=(b._2)
          }
          (a._1 || b._1, buffer)
      }.map(e => e._2.toArray)
  }

  private def kernelizeTrainingSet(data: DataSet[(Boolean, LabeledVector)],
    basisPointsArray: DataSet[Array[Vector]],
    kernel: DistanceMetric): DataSet[(Array[Int], LabeledVector)] = {

    data.mapWithBcVariable(basisPointsArray) {
      (trainingExample, basisPoints) =>
        val newVector = kernelizePoint(trainingExample._2.vector,
          basisPoints, kernel)

        val indices = new ArrayBuffer[Int]

        if (trainingExample._1) {
          val identity = kernel.distance(trainingExample._2.vector, trainingExample._2.vector)
          for (i <- 0 until newVector.size) {
            if (newVector(i) == identity) {
              indices += i
            }
          }
        }
        (indices.toArray, LabeledVector(trainingExample._2.label, newVector))
    }
  }

  private def kernelizePoint(point: Vector,
    basisPointsArray: Array[Vector],
    kernel: DistanceMetric): Vector = {
    val transition = new Array[Double](basisPointsArray.length)

    for (i <- 0 until transition.length) {
      transition(i) = kernel.distance(point, basisPointsArray(i))
    }

    new DenseVector(transition)
  }

  private def batchGradientDescentStep(
    indexedData: DataSet[(Array[Int], LabeledVector)],
    currentWeights: DataSet[WeightVector],
    regularizationConstant: Double,
    learningRate: Double): DataSet[WeightVector] = {

    indexedData.mapWithBcVariable(currentWeights) {
      (indexedPoint, weightVector) =>
        (getRegularizedGradient(indexedPoint,
          weightVector, regularizationConstant), 1)
    }.reduce {
      (left, right) =>
        val (leftGradVector, leftCount) = left
        val (rightGradVector, rightCount) = right
        BLAS.axpy(1.0, leftGradVector.weights, rightGradVector.weights)
        val gradients = WeightVector(
          rightGradVector.weights, leftGradVector.intercept + rightGradVector.intercept)
        (gradients, leftCount + rightCount)
    }.mapWithBcVariableIteration(currentWeights) {
      (gradientCount, weightVector, iteration) =>
        {
          val (WeightVector(weights, intercept), count) = gradientCount

          BLAS.scal(1.0 / count, weights)

          val gradient = WeightVector(weights, intercept / count)
          val effectiveLearningRate = learningRate / Math.sqrt(iteration)

          BLAS.axpy(-effectiveLearningRate, gradient.weights, weightVector.weights)

          WeightVector(
            weightVector.weights,
            weightVector.intercept - effectiveLearningRate * gradient.intercept)
        }
    }
  }

  private def getRegularizedGradient(indexedDataPoint: (Array[Int], LabeledVector),
    weightVector: WeightVector,
    regularizationConstant: Double): WeightVector = {

    val dotRegularization = BLAS.dot(indexedDataPoint._2.vector, weightVector.weights)
    val dot = dotRegularization + weightVector.intercept

    val marginEncroachmentCosts = 1 - dot * indexedDataPoint._2.label
    val marginEncroachments = if (marginEncroachmentCosts > 0) 1 else 0

    val scaler = marginEncroachments * (dot - indexedDataPoint._2.label)
    BLAS.scal(scaler, indexedDataPoint._2.vector)

    for (i <- 0 until indexedDataPoint._1.length) {
      val partialGradientRegularization =
        regularizationConstant * dotRegularization /
          indexedDataPoint._1.length

      indexedDataPoint._2.vector(indexedDataPoint._1(i)) =
        indexedDataPoint._2.vector(indexedDataPoint._1(i)) + partialGradientRegularization
    }
    WeightVector(indexedDataPoint._2.vector, scaler)
  }

  private def getUnregularizedLoss(dataPoint: LabeledVector,
    weightVector: WeightVector): Double = {

    val dot = BLAS.dot(dataPoint.vector, weightVector.weights) +
      weightVector.intercept

    val marginEncroachmentCosts = 1 - dot * dataPoint.label
    val marginEncroachments = if (marginEncroachmentCosts > 0) 1 else 0

    0.5 * marginEncroachments *
      marginEncroachmentCosts * marginEncroachmentCosts
  }
}
