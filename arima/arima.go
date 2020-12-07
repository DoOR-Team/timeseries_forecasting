package arima

import (
	"strconv"

	"github.com/DoOR-Team/goutils/log"
)

func ForeCastARIMA(data []float64, forecastSize int, params Config) *Result {
	p := params.p
	d := params.d
	q := params.q
	P := params.P
	D := params.D
	Q := params.Q
	m := params.m
	paramsForecast := NewConfig(p, d, q, P, D, Q, m)
	paramsXValidation := NewConfig(p, d, q, P, D, Q, m)
	// estimate ARIMA model parameters for forecasting
	fittedModel := estimateARIMA(
		paramsForecast, data, len(data), len(data)+1)

	// compute RMSE to be used in confidence interval computation
	rmseValidation := computeRMSEValidation(
		data, testSetPercentage, paramsXValidation)
	fittedModel.RMSE = rmseValidation

	forecastResult := fittedModel.forecast(forecastSize)

	// populate confidence interval
	forecastResult.SetSigma2AndPredicationInterval(fittedModel.GetParams())

	// add logging messages
	log.Debug("{" +
		"\"Best ModelInterface Param\" : \"" + fittedModel.GetParams().String() + "\"," +
		"\"Forecast Size\" : \"" + strconv.FormatInt(int64(forecastSize), 10) + "\"," +
		"\"Input Size\" : \"" + strconv.FormatInt(int64(len(data)), 10) + "\"" +
		"}")

	// successfully built ARIMA model and its forecast
	return forecastResult
}
