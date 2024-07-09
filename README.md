
# kmeans

k-means clustering algorithm implementation in Go.

## Usage
Only has two functions, `Partition` and `PartitionWithCentroids`.

Use `PartitionWithCentroids` if you want to partition observations with your initial centroids. This can be used if you want to calculate initial centroids with any custom logic.

Use `Partition` if you don't care about the creation of initial centroids and they will be randomly genarated based on observations.

## Example
```go
package main

import (
	"fmt"

	"github.com/mojixcoder/kmeans"
)

// Implements kmeans.Observation
type Location struct {
	Lng, Lat float64
}

func (l Location) X() float64 {
	return l.Lng

}

func (l Location) Y() float64 {
	return l.Lat
}

func main() {
	observations := []kmeans.Observation{
		Location{Lng: 1, Lat: 1},
		Location{Lng: 1, Lat: 2},
		Location{Lng: 2, Lat: 1},
		Location{Lng: 2, Lat: 2},

		Location{Lng: 14, Lat: 14},
		Location{Lng: 15, Lat: 14},
		Location{Lng: 14, Lat: 15},
		Location{Lng: 15, Lat: 15},

		Location{Lng: -10, Lat: -10},
		Location{Lng: -11, Lat: -10},
		Location{Lng: -10, Lat: -11},
		Location{Lng: -11, Lat: -11},
	}

	maxIters := 100

	k := 3

	clusters, err := kmeans.Partition(observations, maxIters, k)
	if err != nil {
		panic(err)
	}

	for _, cluster := range clusters {
		fmt.Println(cluster)
	}
}
```