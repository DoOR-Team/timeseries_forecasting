package arima

import "math"

type Result struct {
	Forecast              []float64
	forecastUpperConf     []float64
	forecastLowerConf     []float64
	dataVariance          float64
	modelRMSE             float64
	maxNormalizedVariance float64
}

func NewResult(pForecast []float64, pDataVariance float64) *Result {

	result := &Result{
		Forecast:              pForecast,
		forecastUpperConf:     make([]float64, len(pForecast)),
		forecastLowerConf:     make([]float64, len(pForecast)),
		dataVariance:          pDataVariance,
		modelRMSE:             -1,
		maxNormalizedVariance: -1,
	}

	// copy(pForecast, result.forecastUpperConf)
	copy(result.forecastLowerConf, pForecast)
	// copy(pForecast, result.forecastUpperConf)
	copy(result.forecastLowerConf, pForecast)
	return result
}

func (r *Result) GetNormalizedVariance(v float64) float64 {
	if v < -0.5 || r.dataVariance < -0.5 {
		return -1
	} else if r.dataVariance < 0.0000001 {
		return v
	} else {
		return math.Abs(v / r.dataVariance)
	}
}

func (r *Result) GetRMSE() float64 {
	return r.modelRMSE
}

func (r *Result) SetRMSE(rmse float64) {
	r.modelRMSE = rmse
}

func (r *Result) GetMaxNormalizedVariance() float64 {
	return r.maxNormalizedVariance
}

func (r *Result) SetConfInterval(constant float64, cumulativeSumOfMA []float64) float64 {
	maxNormalizedVariance := -1.0
	bound := 0.
	for i := 0; i < len(r.Forecast); i++ {
		bound = constant * r.modelRMSE * cumulativeSumOfMA[i]
		r.forecastUpperConf[i] = r.Forecast[i] + bound
		r.forecastLowerConf[i] = r.Forecast[i] - bound
		normalizedVariance := r.GetNormalizedVariance(math.Pow(bound, 2))
		if normalizedVariance > maxNormalizedVariance {
			maxNormalizedVariance = normalizedVariance
		}
	}
	return maxNormalizedVariance
}

func (r *Result) SetSigma2AndPredicationInterval(params Config) {
	r.maxNormalizedVariance = setSigma2AndPredicationInterval(params, r, len(r.Forecast))
}

func (r *Result) GetForecast() []float64 {
	return r.Forecast
}

func (r *Result) GetForecastUpperConf() []float64 {
	return r.forecastUpperConf
}

func (r *Result) GetForecastLowerConf() []float64 {
	return r.forecastLowerConf
}
