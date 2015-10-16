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

import KernelMachinesSampling.KMeansParallel

/**
 * Implements a non linear kernel SVM using a reformulation of the Nyström approximation by
 * Dhruv Mahajan, S. Sathiya Keerthi and S. Sundararajan ("A Distributed Algorithm
 * for Training Nonlinear Kernel Machines", 2014).
 *
 * The kernel has to be positive definite.
 *
 * Labels take their values in {-1; 1}.
 *
 * Basis points are selected using the K-means|| initialization algorithm (B. Bahmani, R. Kumar,
 * B. Moseley, S. Vassilvitskii and A. Vattani. "Scalable K-Means++", 2012)
 *
 * The minimization problem is solved by applying batch gradient descent.
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
 *             // get a positive definite kernel (here, a Gaussian kernel
 *             // with standard deviation = 0.2)
 *             val gaussianKernel = new GaussianDistanceMetric(0.2)
 *
 *             val kernelSVM = KernelSVM()
 *               .setNumberOfBasisPoints(200)
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

  /**
   * Information for the predict operation:
   * selected basis points and learned weight vector
   */
  var basisPoints: Option[DataSet[Array[Vector]]] = None
  var weights: Option[DataSet[Vector]] = None

  /** Information for monitoring */
  var loss: DataSet[Double] = null

  /**
   * Sets the ratio for the approximation
   * of the kernel matrix
   *
   * @param approximationRatio
   * @return itself
   */
  def setNumberOfBasisPoints(numberOfBasisPoints: Int): KernelSVM = {
    parameters.add(NumberOfBasisPoints, numberOfBasisPoints)
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

  case object NumberOfBasisPoints extends Parameter[Int] {
    val defaultValue = Some(10)
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

  def KernalizationOperation(data: DataSet[LabeledVector], numberOfBasisPoints: Int, kernel: DistanceMetric) = {

    val (firstSelection, finalSelection) = KMeansParallel(data,
      numberOfBasisPoints, numberOfBasisPoints, 5)

    val kernelizedData = kernelizeData(firstSelection, finalSelection, kernel)

    kernelizedData
  }

  /**
   * [[FitOperation]] which trains a kernel SVM based on a given training data set.
   *
   */
  implicit val fitKernelSVM = {
    new FitOperation[KernelSVM, LabeledVector] {
      override def fit(
        instance: KernelSVM,
        fitParameters: ParameterMap,
        data: DataSet[LabeledVector]): Unit = {
        val p = instance.parameters ++ fitParameters

        val iterations = p(Iterations)
        val regularizationConstant = p(RegularizationConstant)
        val learningRate = p(LearningRate)

        val kernel = p(Kernel)
        val numberOfBasisPoints = p(NumberOfBasisPoints)

        //////////////////////////////////////////////////////////////
        val (firstSelection, finalSelection) = KMeansParallel(data,
          numberOfBasisPoints, numberOfBasisPoints, 5)

        val kernelizedData = kernelizeData(firstSelection, finalSelection, kernel)

        //////////////////////////////////////////////////////////////

        val weights = kernelizedData.first(1).map(e =>
          WeightVector(DenseVector.init(e._3.vector.size, 0), 0))

        val newWeights = weights.iterate(iterations) {
          weightVectorDS =>
            batchGradientDescentStep(kernelizedData, weightVectorDS,
              regularizationConstant, learningRate)
        }

        // Stores the basis points
        instance.basisPoints = Some(finalSelection.map(e => e._1))
        // Stores the weights
        instance.weights = Some(newWeights.map(e => e.weights))

        // sets the loss for this model in case we want to
        // try out different settings
        instance.loss = kernelizedData.mapWithBcVariable(newWeights) {
          (point, weights) =>
            (getUnregularizedLoss(point._3, weights), 1)
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
  implicit def predictOnVector = {
    new PredictDataSetOperation[KernelSVM, Vector, (Double, Vector)] {
      override def predictDataSet(
        instance: KernelSVM,
        predictParameters: ParameterMap,
        data: DataSet[Vector]): DataSet[(Double, Vector)] = {

        val p = instance.parameters ++ predictParameters

        val kernel = p(Kernel)

        instance.basisPoints match {

          case Some(bps) => {

            val kernelizedData = data.mapWithBcVariable(bps) {
              (point, bp) => kernelizePoint(point, bp, kernel)
            }

            instance.weights match {

              case Some(information) => {
                kernelizedData.mapWithBcVariable(information) {
                  (kernelizedPoint, info) =>

                    val dot = BLAS.dot(kernelizedPoint, info)
                    val prediction = if (dot > 0) 1 else -1
                    (prediction, kernelizedPoint)
                }
              }
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
  implicit def predictOnLabeledVector = {
    new PredictDataSetOperation[KernelSVM, LabeledVector, (Double, LabeledVector)] {
      override def predictDataSet(
        instance: KernelSVM,
        predictParameters: ParameterMap,
        data: DataSet[LabeledVector]): DataSet[(Double, LabeledVector)] = {

        val p = instance.parameters ++ predictParameters

        val kernel = p(Kernel)

        instance.basisPoints match {

          case Some(bps) => {

            val kernelizedData = data.mapWithBcVariable(bps) {
              (point, bp) => LabeledVector(point.label, kernelizePoint(point.vector, bp, kernel))
            }

            instance.weights match {

              case Some(information) => {
                kernelizedData.mapWithBcVariable(information) {
                  (kernelizedPoint, info) =>

                    val dot = BLAS.dot(kernelizedPoint.vector, info)
                    val prediction = if (dot > 0) 1 else -1
                    (prediction, kernelizedPoint)
                }
              }
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

  /*
   * Each training example is transposed into the feature space.
   * The first Int is the number of points like self (self included)
   * that correspond to a basis point
   * The second Int is the index where a point has an impact on
   * the regularization
   */
  private def kernelizeData(firstSelection: DataSet[(Boolean, LabeledVector)],
    finalSelection: DataSet[(Array[Vector], Array[Int])],
    kernel: DistanceMetric): DataSet[(Int, Int, LabeledVector)] = {
    firstSelection.mapWithBcVariable(finalSelection) {
      (point, centers) =>
        if (point._1) {
          val index = centers._1.indexOf(point._2.vector)
          if (index == -1) {
            (0, 0, LabeledVector(point._2.label, kernelizePoint(point._2.vector, centers._1, kernel)))
          } else {
            (centers._2(index), point._2)
            (centers._2(index), index, LabeledVector(point._2.label, kernelizePoint(point._2.vector, centers._1, kernel)))
          }
        } else {
          (0, 0, LabeledVector(point._2.label, kernelizePoint(point._2.vector, centers._1, kernel)))
        }
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
    indexedData: DataSet[(Int, Int, LabeledVector)],
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

  private def getRegularizedGradient(indexedDataPoint: (Int, Int, LabeledVector),
    weightVector: WeightVector,
    regularizationConstant: Double): WeightVector = {

    val dotRegularization = BLAS.dot(indexedDataPoint._3.vector, weightVector.weights)
    val dot = dotRegularization + weightVector.intercept

    val marginEncroachmentCosts = 1 - dot * indexedDataPoint._3.label
    val marginEncroachments = if (marginEncroachmentCosts > 0) 1 else 0

    val scaler = marginEncroachments * (dot - indexedDataPoint._3.label)
    BLAS.scal(scaler, indexedDataPoint._3.vector)

    // for (i <- 0 until indexedDataPoint._1.length) 

    if (indexedDataPoint._1 > 0) {
      val partialGradientRegularization =
        regularizationConstant * dotRegularization /
          indexedDataPoint._1

      indexedDataPoint._3.vector(indexedDataPoint._2) =
        indexedDataPoint._3.vector(indexedDataPoint._2) + partialGradientRegularization
    }
    WeightVector(indexedDataPoint._3.vector, scaler)
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
