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

import org.apache.flink.ml.common._
import org.apache.flink.api.scala._
import scala.collection.mutable.ArrayBuffer
import org.apache.flink.ml.math._

object KernelMachinesSampling {

  def KMeansParallel(data: DataSet[LabeledVector],
    numberOfCenters: Int, oversamplingFactor: Int, iterations: Int)
  : (DataSet[(Boolean, LabeledVector)], DataSet[(Array[Vector], Array[Int])]) = {

    val initialState = getInitialState(data)

    val firstSelection = initialState.iterate(iterations) {
      state =>
        val info = getInformation(state)
        chooseNewCenters(state, info, oversamplingFactor)
    }.map(e => (e._2 == 0., e._3))

    val cleanIntermediateState = firstSelection.filter {
      e => e._1
    }.map {
      e => e._2.vector
    }.reduceGroup {
      point =>
        val basisPoints = new ArrayBuffer[Vector]
        val occurrences = new ArrayBuffer[Int]
        while (point.hasNext) {
          // we make sure there is no duplicate among the selection.
          // If there are, we delete the duplicates and keep track
          // of the number of duplicates.
          // This way, we can map the result of the second clustering
          // onto the first selection, telling to each to what extend
          // it has to have an impact.
          val newPoint = point.next
          val duplicate = basisPoints.indexOf(newPoint)
          if (duplicate == -1) {
            basisPoints += newPoint
            occurrences += 1
          } else {
            occurrences(duplicate) = occurrences(duplicate) + 1
          }
        }
        (basisPoints.toArray, occurrences.toArray)
    }

    // we cluster the centers on a single machine
    val finalSelection = cleanIntermediateState.map {
      e => kMeansPlusPlus(e, numberOfCenters)
    }
    
    (firstSelection, finalSelection)
  }

  def kMeansPlusPlus(centersToCluster: (Array[Vector], Array[Int]), WishedNumberOfCenters: Int): (Array[Vector], Array[Int]) = {
    val numberOfCenters = min(WishedNumberOfCenters, centersToCluster._1.length).toInt
    val centers = new Array[Vector](numberOfCenters)
    val centerCorrespondances = new Array[Int](numberOfCenters)
    val costs = new Array[Double](centersToCluster._1.length)

    // chooses a first center
    val index = scala.util.Random.nextInt(centersToCluster._1.length)
    centers(0) = centersToCluster._1(index)
    centerCorrespondances(0) = centersToCluster._2(index)

    var totalCost = 0.
    var k = 0
    var threshold = 0.

    for (i <- 1 until min(numberOfCenters, 2).toInt) {

      // compute the costs relative to the first center
      totalCost = 0.
      for (j <- 0 until costs.length) {
        costs(j) = getSquaredDistance(centersToCluster._1(j), centers(i - 1))
        totalCost = totalCost + costs(j)
      }

      // choose a new center
      threshold = scala.util.Random.nextDouble * totalCost
      k = -1
      while (threshold > 0) {
        k = k + 1
        threshold = threshold - costs(k)
      }
      centers(i) = centersToCluster._1(k)
      centerCorrespondances(i) = centersToCluster._2(k)
    }

    for (i <- 2 until numberOfCenters) {

      // compute the costs relative to the last new center
      totalCost = 0.
      for (j <- 0 until costs.length) {
        costs(j) = min(costs(j), getSquaredDistance(centersToCluster._1(j), centers(i - 1)))
        totalCost = totalCost + costs(j)
      }

      // choose a new center
      threshold = scala.util.Random.nextDouble * totalCost
      k = -1
      while (threshold > 0) {
        k = k + 1
        threshold = threshold - costs(k)
      }
      centers(i) = centersToCluster._1(k)
      centerCorrespondances(i) = centersToCluster._2(k)
    }
    (centers, centerCorrespondances)
  }

  // ========================================== Auxiliary functions ================================

  // the first center is arbitrarily the first element given from the data
  private def getInitialState(data: DataSet[LabeledVector]): DataSet[(Boolean, Double, LabeledVector)] = {
    val firstCenter = data.first(1).map(e => e.vector)

    data.mapWithBcVariable(firstCenter) {
      (point, center) =>
        val squaredDistance = getSquaredDistance(point.vector, center)
        (squaredDistance == 0.0, squaredDistance, point)
    }
  }

  // computes the total cost and extracts the new centers
  private def getInformation(
    markedData: DataSet[(Boolean, Double, LabeledVector)]): DataSet[(Double, Array[Vector])] = {
    markedData.map {
      e =>
        val center = new ArrayBuffer[Vector]
        if (e._1) {
          center += e._3.vector
        }
        (e._2, center)
    }.reduce {
      (e1, e2) =>
        (e1._1 + e2._1, e1._2 ++= e2._2)
    }.map {
      e => (e._1, e._2.toArray)
    }
  }

  // computes the new distribution, selects the new centers and updates the local costs
  private def chooseNewCenters(
    markedData: DataSet[(Boolean, Double, LabeledVector)],
    information: DataSet[(Double, Array[Vector])],
    oversamplingFactor: Int): DataSet[(Boolean, Double, LabeledVector)] = {
    markedData.mapWithBcVariable(information) {
      (point, info) =>
        val squaredMinimalDistance = {
          // in case no new center has been chosen
          if (info._2.length > 0) {
            min(point._2,
              getSquaredMinimalDistance(point._3.vector, info._2))
          } else {
            point._2
          }
        }
        val probability = oversamplingFactor * squaredMinimalDistance / info._1
        val isNewCenter = scala.util.Random.nextDouble < probability
        val cost = if (isNewCenter) 0 else squaredMinimalDistance
        (isNewCenter, cost, point._3)
    }
  }

  private def getSquaredMinimalDistance(point: Vector,
    centers: Array[Vector]): Double = {
    var minimalSquaredDistance = getSquaredDistance(point, centers(0))
    for (i <- 1 until centers.size) {
      val currentSquaredDistance = getSquaredDistance(point, centers(i))
      minimalSquaredDistance = min(minimalSquaredDistance, currentSquaredDistance)
    }
    minimalSquaredDistance
  }

  private def getSquaredDistance(point1: Vector, point2: Vector): Double = {
    var result = 0.
    for (i <- 0 until point1.size) {
      val diff = point1(i) - point2(i)
      result = result + diff * diff
    }
    result
  }

  private def min(a: Double, b: Double): Double = {
    if (a < b) a else b
  }

}
