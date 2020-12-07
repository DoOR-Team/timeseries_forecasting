package arima

type Model struct {
	Params        Config
	data          []float64
	trainDataSize int
	RMSE          float64
	solver        *Solver
}

// func (m *Model) ForecastResult(data []float64, forecastSize int) []float64 {
// 	fittedModel := estimateARIMA(
// 		m, data, len(data), len(data)+1)
//
// 	paramsXValidation := m.Config
// 	rmseValidation := computeRMSEValidation(
// 		m.Config, testSetPercentage, paramsXValidation)
//
// 	forecastResult := fittedModel.forecast(forecastSize)
// 	// forecastResult.setSigma2AndPredicationInterval(m)
// 	return forecastResult
// }

func (m *Model) forecast(forecastSize int) *Result {
	forecastResult := forecastARIMA(m.Params, m.data, m.trainDataSize, m.trainDataSize+forecastSize)
	forecastResult.modelRMSE = m.RMSE
	// forecastResult.setSigma2AndPredicationInterval(m)
	return forecastResult
}

func (m *Model) GetParams() Config {
	return m.Params
}
