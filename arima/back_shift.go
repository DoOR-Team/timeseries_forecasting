package arima

import (
	"fmt"

	"github.com/DoOR-Team/goutils/log"
)

type BackShift struct {
	_degree  int // maximum lag, e.g. AR(1) degree will be 1
	_indices []bool
	_offsets []int
	_coeffs  []float64
}

func NewBackShift(degree int, initial bool) *BackShift {
	if degree < 0 {
		log.Fatal("degree must be non-negative")
	}
	b := &BackShift{
		_degree:  degree,
		_indices: make([]bool, degree+1),
		_offsets: nil,
		_coeffs:  nil,
	}
	for j := 0; j <= degree; j++ {
		b._indices[j] = initial
	}
	b._indices[0] = true // zero index must be true all the time
	return b
}

func (b *BackShift) getDegree() int {
	return b._degree
}

func (b *BackShift) getCoefficientsFlattened() []float64 {
	if b._degree <= 0 || b._offsets == nil || b._coeffs == nil {
		return make([]float64, 0)
	}
	temp := -1
	for _, offset := range b._offsets {
		if offset > temp {
			temp = offset
		}
	}
	maxIdx := 1 + temp
	flattened := make([]float64, maxIdx)
	for j := 0; j < maxIdx; j++ {
		flattened[j] = 0
	}
	for j := 0; j < len(b._offsets); j++ {
		flattened[b._offsets[j]] = b._coeffs[j]
	}
	return flattened
}
func (b *BackShift) setIndex(index int, enable bool) {
	b._indices[index] = enable
}

func (b *BackShift) apply(another *BackShift) *BackShift {
	mergedDegree := b._degree + another._degree
	merged := make([]bool, mergedDegree+1)
	for j := 0; j <= mergedDegree; j++ {
		merged[j] = false
	}
	for j := 0; j <= b._degree; j++ {
		if b._indices[j] {
			for k := 0; k <= another._degree; k++ {
				merged[j+k] = merged[j+k] || another._indices[k]
			}
		}
	}
	return &BackShift{
		_degree:  mergedDegree,
		_indices: merged,
		_offsets: nil,
		_coeffs:  nil,
	}
}

func (b *BackShift) initializeParams(includeZero bool) {
	b._indices[0] = includeZero
	b._offsets = nil
	b._coeffs = nil
	nonzeroCount := 0
	for j := 0; j <= b._degree; j++ {
		if b._indices[j] {
			nonzeroCount++
		}
	}
	b._offsets = make([]int, nonzeroCount) // cannot be 0 as 0-th index is always true
	b._coeffs = make([]float64, nonzeroCount)
	coeffIndex := 0
	for j := 0; j <= b._degree; j++ {
		if b._indices[j] {
			b._offsets[coeffIndex] = j
			b._coeffs[coeffIndex] = 0
			coeffIndex++
		}
	}
}

// MAKE SURE to initializeParams before calling below methods
func (b *BackShift) numParams() int {
	return len(b._offsets)
}

func (b *BackShift) paramOffsets() []int {
	return b._offsets
}

func (b *BackShift) getParam(paramIndex int) float64 {
	for j := 0; j < len(b._offsets); j++ {
		if b._offsets[j] == paramIndex {
			return b._coeffs[j]
		}
	}
	panic(fmt.Sprintf("invalid parameter index: %d", paramIndex))
}

func (b *BackShift) getAllParam() []float64 {
	return b._coeffs
}

func (b *BackShift) setParam(paramIndex int, paramValue float64) {
	offsetIndex := -1
	for j := 0; j < len(b._offsets); j++ {
		if b._offsets[j] == paramIndex {
			offsetIndex = j
			break
		}
	}
	if offsetIndex == -1 {
		panic(fmt.Sprintf("invalid parameter index: %d", paramIndex))
	}
	b._coeffs[offsetIndex] = paramValue
}

func (b *BackShift) copyParamsToArray(dest []float64) {
	// copy(b._coeffs, dest)
	copy(dest, b._coeffs)
}

func (b *BackShift) getLinearCombinationFrom(timeseries []float64, tsOffset int) float64 {
	linearSum := 0.
	for j := 0; j < len(b._offsets); j++ {
		linearSum += timeseries[tsOffset-b._offsets[j]] * b._coeffs[j]
	}
	return linearSum
}
