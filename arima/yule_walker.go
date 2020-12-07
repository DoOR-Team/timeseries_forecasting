package arima

import (
	"math"

	"github.com/DoOR-Team/goutils/log"
	"github.com/DoOR-Team/timeseries_forecasting/arima/matrix"
)

func Fit(data []float64, p int) []float64 {

	length := len(data)
	if length == 0 || p < 1 {
		log.Fatalf(
			"fitYuleWalker - Invalid Parameters length= %d, p ", length, p)
	}

	r := make([]float64, p+1)
	for _, aData := range data {
		r[0] += math.Pow(aData, 2)
	}
	r[0] /= float64(length)

	for j := 1; j < p+1; j++ {
		for i := 0; i < length-j; i++ {
			r[j] += data[i] * data[i+j]
		}
		r[j] /= float64(length)
	}

	toeplitz := initToeplitz(r[0:p])
	rVector := matrix.NewInsightVectorWithData(r[1:p+1], false)

	return toeplitz.SolveSPDIntoVector(rVector, maxConditionNumber).DeepCopy()
}
