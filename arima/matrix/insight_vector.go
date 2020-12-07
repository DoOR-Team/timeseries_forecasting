package matrix

import "fmt"

type InsightsVector struct {
	_m     int
	_data  []float64
	_valid bool
}

func NewInsightVector(m int, value float64) InsightsVector {
	vector := InsightsVector{
		_m:     -1,
		_data:  nil,
		_valid: false,
	}
	if m <= 0 {
		panic("[InsightsVector] invalid size")
	} else {
		vector._data = make([]float64, m)
		for j := 0; j < m; j++ {
			vector._data[j] = value
		}
		vector._m = m
		vector._valid = true
	}
	return vector
}

func NewInsightVectorWithData(data []float64, deepCopy bool) *InsightsVector {
	vector := &InsightsVector{
		_m:     -1,
		_data:  nil,
		_valid: false,
	}
	if data == nil || len(data) == 0 {
		panic("[InsightsVector] invalid data")
	} else {
		vector._m = len(data)
		if deepCopy {
			vector._data = make([]float64, len(data))
			// copy(vector._data, data)
			copy(data, vector._data)
		} else {
			vector._data = data
		}
		vector._valid = true
	}
	return vector
}

func (i *InsightsVector) DeepCopy() []float64 {
	dataDeepCopy := make([]float64, i._m)
	copy(dataDeepCopy, i._data)
	return dataDeepCopy
}

func (iv *InsightsVector) Get(i int) float64 {
	if !iv._valid {
		panic("[InsightsVector] invalid Vector")
	} else if i >= iv._m {
		panic(fmt.Sprintf("[InsightsVector] Index: %d, Size: %d", i, iv._m))
	}
	return iv._data[i]
}

func (iv *InsightsVector) Size() int {
	if !iv._valid {
		panic("[InsightsVector] invalid Vector")
	}

	return iv._m
}

func (iv *InsightsVector) Set(i int, val float64) {
	if !iv._valid {
		panic("[InsightsVector] invalid Vector")
	} else if i >= iv._m {
		panic(
			fmt.Sprintf("[InsightsVector] Index: %d, Size: %d", i, iv._m))
	}
	iv._data[i] = val
}

func (iv *InsightsVector) Dot(vector *InsightsVector) float64 {
	if !iv._valid || !vector._valid {
		panic("[InsightsVector] invalid Vector")
	} else if iv._m != vector.Size() {
		panic("[InsightsVector][dot] invalid vector size.")
	}

	sumOfProducts := 0.
	for i := 0; i < iv._m; i++ {
		sumOfProducts += iv._data[i] * vector.Get(i)
	}
	return sumOfProducts
}
