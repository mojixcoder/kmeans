package kmeans

import (
	"errors"
	"math"
	"math/rand"
	"slices"
)

type (
	// Observation is the observations data type.
	Observation interface {
		X() float64
		Y() float64
	}

	// centroid is used for re-calculating centroids.
	centroid struct {
		x, y float64
	}
)

// X returns centroid's X.
func (c centroid) X() float64 {
	return c.x
}

// Y returns centroid's Y.
func (c centroid) Y() float64 {
	return c.y
}

// getCentroids calculates k random centroids from the observations.
func getCentroids(observations []Observation, k int) ([]Observation, error) {
	if len(observations) < k {
		return nil, errors.New("not enough data to calculate centroids")
	}

	if k == 0 {
		return nil, errors.New("k should be greater than 0")
	}

	cloneObservations := make([]Observation, len(observations))

	rand.Shuffle(len(cloneObservations), func(i, j int) {
		cloneObservations[i] = observations[j]
		cloneObservations[j] = observations[i]
	})

	return slices.Clip(cloneObservations[:k]), nil
}

// euclideanDistane calculates euclidean distance between two points.
func euclideanDistane(p1, p2 Observation) float64 {
	return math.Sqrt(math.Pow(p2.X()-p1.X(), 2) + math.Pow(p2.Y()-p1.Y(), 2))
}

// minCentroidIdx finds the nearest centroids index to the point.
func minCentroidIdx(p Observation, centroids []Observation) int {
	minIdx := -1
	var minDist float64 = -1

	for i, centroid := range centroids {
		dist := euclideanDistane(p, centroid)
		if dist < minDist || i == 0 {
			minIdx = i
			minDist = dist
		}
	}

	return minIdx
}

func recalculateCentroid(cluster []Observation) Observation {
	var sumX, sumY float64

	for _, p := range cluster {
		sumX += p.X()
		sumY += p.Y()
	}

	return centroid{x: sumX / float64(len(cluster)), y: sumY / float64(len(cluster))}
}

func centroidsAreEqual(c1, c2 []Observation) bool {
	if len(c1) != len(c2) {
		return false
	}

	for i := 0; i < len(c1); i++ {
		if c1[i].X() != c2[i].X() || c1[i].Y() != c2[i].Y() {
			return false
		}
	}

	return true
}

// PartitionWithCentroids partitions observations with the given initial centroids.
//
// This can be used if you want to calculate initial centroids with any custom logic.
func PartitionWithCentroids(observations, centroids []Observation, maxIters int) ([][]Observation, error) {
	if len(centroids) == 0 || len(observations) == 0 || maxIters == 0 {
		return nil, errors.New("empty value for observations or centroids or maxIters")
	}

	var isLastIter bool
	finalCusters := make([][]Observation, 0)

	for i := 0; i < maxIters; i++ {
		clusters := make([][]Observation, len(centroids))

		for _, p := range observations {
			idx := minCentroidIdx(p, centroids)
			clusters[idx] = append(clusters[idx], p)
		}

		finalCusters = clusters

		if isLastIter {
			break
		}

		prevCentroids := slices.Clone(centroids)

		for i, cluster := range clusters {
			if len(cluster) > 0 {
				centroids[i] = recalculateCentroid(cluster)
			}
		}

		// Centroids didn't change.
		if centroidsAreEqual(prevCentroids, centroids) {
			isLastIter = true
		}
	}

	return finalCusters, nil
}

// Partition partitions observations with random centroids selected from the observations.
func Partition(observations []Observation, maxIters, k int) ([][]Observation, error) {
	centroids, err := getCentroids(observations, k)
	if err != nil {
		return nil, err
	}

	return PartitionWithCentroids(observations, centroids, maxIters)
}
