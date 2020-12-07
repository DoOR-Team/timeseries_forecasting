package utils

import "fmt"

func Differentiate(src, dst, initial []float64, d int) {
	if initial == nil || len(initial) != d || d <= 0 {
		panic(fmt.Sprintf("invalid initial size=%d, d=%d", len(initial), d))
	}
	if src == nil || len(src) <= d {
		panic(fmt.Sprintf("insufficient source size=%d, d=%d", len(src), d))
	}
	if dst == nil || len(dst) != len(src)-d {
		panic(fmt.Sprintf(
			"invalid destination size=%d, src=%d, d=%d", len(dst), len(src), d))
	}

	// copy over initial conditions
	// copy(src, initial)
	copy(initial, src)
	// now differentiate source into destination
	src_len := len(src)
	k := 0
	for j := d; j < src_len; j++ {
		dst[k] = src[j] - src[k]
		k++
	}

}

func Integrate(src, dst, initial []float64, d int) {
	if initial == nil || len(initial) != d || d <= 0 {
		panic(fmt.Sprintf("invalid initial size=%d, d=%d", len(initial), d))
	}
	if src == nil || len(src) <= d {
		panic(fmt.Sprintf("insufficient source size=%d, d=%d", len(src), d))
	}
	if dst == nil || len(src) != len(dst)-d {
		panic(fmt.Sprintf(
			"invalid destination size=%d, src=%d, d=%d", len(dst), len(src), d))
	}

	// copy over initial conditions
	// copy(initial, dst)
	copy(dst, initial)
	// now integrate source into destination
	src_len := len(src)
	k := 0
	for j := d; k < src_len; j++ {
		dst[j] = dst[k] + src[k]
		k++
	}
}

func Shift(inputData []float64, shiftAmount float64) {
	for i := 0; i < len(inputData); i++ {
		inputData[i] += shiftAmount
	}
}

func ComputeMean(data []float64) float64 {
	length := len(data)
	if length == 0 {
		return 0.0
	}
	sum := 0.0
	for i := 0; i < length; i++ {
		sum += data[i]
	}
	return sum / float64(length)
}

func ComputeVariance(data []float64) float64 {
	variance := 0.0
	mean := ComputeMean(data)
	for i := 0; i < len(data); i++ {
		diff := data[i] - mean
		variance += diff * diff
	}
	return variance / float64(len(data)-1)
}
