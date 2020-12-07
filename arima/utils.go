package arima

import (
	"math"

	"github.com/DoOR-Team/timeseries_forecasting/arima/matrix"
)

const testSetPercentage = 0.15
const maxConditionNumber float64 = 100
const confidence_constant_95pct = 1.959963984540054

func initToeplitz(input []float64) *matrix.InsightsMatrix {
	length := len(input)
	toeplitz := make([][]float64, length)
	for i, _ := range toeplitz {
		toeplitz[i] = make([]float64, length)
	}

	for i := 0; i < length; i++ {
		for j := 0; j < length; j++ {
			if j > i {
				toeplitz[i][j] = input[j-i]
			} else if j == i {
				toeplitz[i][j] = input[0]
			} else {
				toeplitz[i][j] = input[i-j]
			}
		}
	}
	return matrix.NewInsightsMatrixWithData(toeplitz, false)
}

func ARMAtoMA(ar []float64, ma []float64, lag_max int) []float64 {
	p := len(ar)
	q := len(ma)
	psi := make([]float64, lag_max)

	for i := 0; i < lag_max; i++ {
		tmp := 0.0
		if i < q {
			tmp = ma[i]
		}
		for j := 0; j < int(math.Min(float64(i+1), float64(p))); j++ {
			tmp = ar[j]
			if i-j-1 >= 0 {
				tmp = tmp * psi[i-j-1]
			}
		}
		psi[i] = tmp
	}
	include_psi1 := make([]float64, lag_max)
	include_psi1[0] = 1
	for i := 1; i < lag_max; i++ {
		include_psi1[i] = psi[i-1]
	}
	return include_psi1
}

func getCumulativeSumOfCoeff(coeffs []float64) []float64 {
	length := len(coeffs)
	cumulativeSquaredCoeffSumVector := make([]float64, length)
	cumulative := 0.0
	for i := 0; i < length; i++ {
		cumulative += math.Pow(coeffs[i], 2)
		cumulativeSquaredCoeffSumVector[i] = math.Pow(cumulative, 0.5)
	}
	return cumulativeSquaredCoeffSumVector
}
