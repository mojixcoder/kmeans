package kmeans

import (
	"fmt"
	"slices"
	"testing"
)

func TestGetCentroids(t *testing.T) {
	testCases := []struct {
		observations    []Observation
		k               int
		hasErr          bool
		observationsLen int
	}{
		{observations: nil, k: 1, hasErr: true, observationsLen: 0},
		{observations: nil, k: 0, hasErr: true, observationsLen: 0},
		{observations: []Observation{centroid{1, 1}, centroid{2, 2}}, k: 2, hasErr: false, observationsLen: 2},
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			res, err := getCentroids(tc.observations, tc.k)

			if tc.hasErr {
				if err == nil {
					t.Errorf("expected error not to be nil, err: %s", err)
				}
			} else {
				if err != nil {
					t.Errorf("expected error to be nil, err: %s", err)
				}
			}

			if len(res) != tc.observationsLen {
				t.Errorf(
					"length of centroids is not equal to the expected length, %d != %d",
					tc.observationsLen,
					len(res),
				)
			}
		})
	}
}

func TestEuclideanDistane(t *testing.T) {
	dist := euclideanDistane(centroid{0, 1}, centroid{2, 1})

	if dist != 2 {
		t.Errorf("invalid euclidean distane calculated, %f", dist)
	}
}

func TestMinCentroidIdx(t *testing.T) {
	idx := minCentroidIdx(centroid{0, 0}, []Observation{
		centroid{10, 10},
		centroid{-10, 10},
		centroid{1, 1},
		centroid{2, 2},
	})

	if idx != 2 {
		t.Errorf("invalid index calculated, %d", idx)
	}
}

func TestRecalculateCentroid(t *testing.T) {
	centroid := recalculateCentroid([]Observation{
		centroid{1, 2},
		centroid{1, 2},
		centroid{1, 2},
		centroid{1, 2},
	})

	if centroid.X() != 1 {
		t.Errorf("invalid x calculated for new centroid, %f", centroid.X())
	}

	if centroid.Y() != 2 {
		t.Errorf("invalid y calculated for new centroid, %f", centroid.Y())
	}
}

func TestCentroidsAreEqual(t *testing.T) {
	testCases := []struct {
		expectedRes bool
		c1, c2      []Observation
	}{
		{c1: nil, c2: []Observation{centroid{}}, expectedRes: false},
		{c1: []Observation{centroid{1, 1}}, c2: []Observation{centroid{2, 2}}, expectedRes: false},
		{c1: []Observation{centroid{1, 1}}, c2: []Observation{centroid{1, 1}}, expectedRes: true},
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			res := centroidsAreEqual(tc.c1, tc.c2)

			if res != tc.expectedRes {
				t.Errorf("invalid result, expected %t but got %t", tc.expectedRes, res)
			}
		})
	}
}

func TestPartitionWithCentroids(t *testing.T) {
	_, err := PartitionWithCentroids(nil, nil, 0)
	if err == nil {
		t.Errorf("expected error not to be nil, err: %s", err)
	}

	observations := []Observation{
		centroid{x: 1, y: 1},
		centroid{x: 1, y: 2},
		centroid{x: 2, y: 1},
		centroid{x: 2, y: 2},

		centroid{x: 14, y: 14},
		centroid{x: 15, y: 14},
		centroid{x: 14, y: 15},
		centroid{x: 15, y: 15},

		centroid{x: -10, y: -10},
		centroid{x: -11, y: -10},
		centroid{x: -10, y: -11},
		centroid{x: -11, y: -11},
	}

	centroids := []Observation{
		centroid{x: 1, y: 1},
		centroid{x: 14, y: 14},
		centroid{x: -10, y: -10},
	}

	clusters, err := PartitionWithCentroids(observations, centroids, 10)

	if err != nil {
		t.Errorf("expected error to be nil, err: %s", err)
	}

	for i := range clusters {
		if !slices.Equal(clusters[i], observations[i*4:i*4+4]) {
			t.Errorf(
				"expected cluster[%d] to be equal to observations[%d:%d], %v --- %v",
				i, i*4, i*4+4, clusters[0], observations[0:4],
			)
		}
	}
}

func TestPartition(t *testing.T) {
	_, err := Partition(nil, 10, 1)
	if err == nil {
		t.Errorf("expected error not to be nil, err: %s", err)
	}

	observations := []Observation{
		centroid{x: 1, y: 1},
		centroid{x: 1, y: 2},
		centroid{x: 2, y: 1},
		centroid{x: 2, y: 2},

		centroid{x: 14, y: 14},
		centroid{x: 15, y: 14},
		centroid{x: 14, y: 15},
		centroid{x: 15, y: 15},

		centroid{x: -10, y: -10},
		centroid{x: -11, y: -10},
		centroid{x: -10, y: -11},
		centroid{x: -11, y: -11},
	}

	clusters, err := Partition(observations, 100, 3)

	if err != nil {
		t.Errorf("expected error to be nil, err: %s", err)
	}

	if len(clusters) != 3 {
		t.Errorf("invalid clusters, %v", clusters)
	}
}
