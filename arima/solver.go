package arima

import (
	"math"

	"github.com/DoOR-Team/goutils/log"
	mtx "github.com/DoOR-Team/timeseries_forecasting/arima/matrix"
	"github.com/DoOR-Team/timeseries_forecasting/arima/utils"
)

const maxIterationForHannanRissanen = 5

type Solver struct {
}

func newSolver() *Solver {
	return &Solver{}
}

func forecastARMA(params Config, dataStationary []float64, startIndex int, endIndex int) []float64 {
	trainLen := startIndex
	totalLen := endIndex
	errors := make([]float64, totalLen)
	data := make([]float64, totalLen)
	// copy(dataStationary, data)
	copy(data, dataStationary)
	forecastLen := endIndex - startIndex
	forecasts := make([]float64, forecastLen)
	_dp := params.getDegreeP()
	_dq := params.getDegreeQ()
	startIdx := int(math.Max(float64(_dp), float64(_dq)))

	for j := 0; j < startIndex; j++ {
		errors[j] = 0
	}

	// populate errors and forecasts
	for j := startIdx; j < trainLen; j++ {
		forecast := params.forecastOnePointARMA(data, errors, j)
		dataError := data[j] - forecast
		errors[j] = dataError
	}
	// now we can forecast
	for j := trainLen; j < totalLen; j++ {
		forecast := params.forecastOnePointARMA(data, errors, j)
		data[j] = forecast
		errors[j] = 0
		forecasts[j-trainLen] = forecast
	}
	// return forecasted values
	return forecasts
}

func forecastARIMA(params Config, data []float64, forecastStartIndex int, forecastEndIndex int) *Result {
	if !checkARIMADataLength(params, data, forecastStartIndex, forecastEndIndex) {
		initialConditionSize := params.d + params.D*params.m
		log.Fatalf(
			"not enough data for ARIMA. needed at least %d, have %d, startindex=%d, endindex = %d", initialConditionSize,
			len(data), forecastStartIndex, forecastEndIndex)
	}

	forecast_length := forecastEndIndex - forecastStartIndex
	forecast := make([]float64, forecast_length)
	data_train := make([]float64, forecastStartIndex)
	// copy(data, data_train)
	copy(data_train, data)

	// DIFFERENTIATE
	hasSeasonalI := params.D > 0 && params.m > 0
	hasNonSeasonalI := params.d > 0
	data_stationary := differentiate(params, data_train, hasSeasonalI,
		hasNonSeasonalI) // currently un-centered

	// END OF DIFFERENTIATE
	// ==========================================

	// =========== CENTERING ====================
	mean_stationary := utils.ComputeMean(data_stationary)
	utils.Shift(data_stationary, (-1)*mean_stationary)
	dataVariance := utils.ComputeVariance(data_stationary)
	// ==========================================

	// ==========================================
	// FORECAST
	forecast_stationary := forecastARMA(params, data_stationary,
		len(data_stationary),
		len(data_stationary)+forecast_length)

	data_forecast_stationary := make([]float64, len(data_stationary)+forecast_length)
	// copy(data_stationary, data_forecast_stationary)
	copy(data_forecast_stationary, data_stationary)
	// copy(forecast_stationary, data_forecast_stationary)
	copy(data_forecast_stationary[len(data_stationary):], forecast_stationary)

	// END OF FORECAST
	// ==========================================

	// =========== UN-CENTERING =================
	utils.Shift(data_forecast_stationary, mean_stationary)
	// ==========================================

	// ===========================================
	// INTEGRATE
	forecast_merged := integrate(params, data_forecast_stationary, hasSeasonalI,
		hasNonSeasonalI)
	// END OF INTEGRATE
	// ===========================================
	copy(forecast, forecast_merged[forecastStartIndex:])

	return NewResult(forecast, dataVariance)
}

func estimateARIMA(params Config, data []float64, forecastStartIndex int, forecastEndIndex int) *Model {
	if !checkARIMADataLength(params, data, forecastStartIndex, forecastEndIndex) {
		initialConditionSize := params.d + params.D*params.m
		log.Fatalf(
			"not enough data for ARIMA. needed at least %d, have %d, startindex=%d, endindex = %d", initialConditionSize,
			len(data), forecastStartIndex, forecastEndIndex)
	}
	forecast_length := forecastEndIndex - forecastStartIndex
	data_train := make([]float64, forecastStartIndex)
	// copy(data, data_train)
	copy(data_train, data)
	hasSeasonalI := params.D > 0 && params.m > 0
	hasNonSeasonalI := params.d > 0
	data_stationary := differentiate(params, data_train, hasSeasonalI,
		hasNonSeasonalI) // currently un-centered
	// END OF DIFFERENTIATE
	// ==========================================

	// =========== CENTERING ====================
	mean_stationary := utils.ComputeMean(data_stationary)
	utils.Shift(data_stationary, (-1)*mean_stationary)
	// ==========================================
	// FORECAST
	estimateARMA(data_stationary, &params, forecast_length,
		maxIterationForHannanRissanen)

	return &Model{Params: params, data: data, trainDataSize: forecastStartIndex}
}

func differentiate(params Config, trainingData []float64,
	hasSeasonalI bool, hasNonSeasonalI bool) []float64 {
	var dataStationary []float64 // currently un-centered
	if hasSeasonalI && hasNonSeasonalI {
		params.differentiateSeasonal(trainingData)
		params.differentiateNonSeasonal(params.getLastDifferenceSeasonal())
		dataStationary = params.getLastDifferenceNonSeasonal()
	} else if hasSeasonalI {
		params.differentiateSeasonal(trainingData)
		dataStationary = params.getLastDifferenceSeasonal()
	} else if hasNonSeasonalI {
		params.differentiateNonSeasonal(trainingData)
		dataStationary = params.getLastDifferenceNonSeasonal()
	} else {
		dataStationary = make([]float64, len(trainingData))
		// copy(trainingData, dataStationary)
		copy(dataStationary, trainingData)
	}

	return dataStationary
}

func integrate(params Config, dataForecastStationary []float64,
	hasSeasonalI bool, hasNonSeasonalI bool) []float64 {
	var forecastMerged []float64
	if hasSeasonalI && hasNonSeasonalI {
		params.integrateSeasonal(dataForecastStationary)
		params.integrateNonSeasonal(params.getLastIntegrateSeasonal())
		forecastMerged = params.getLastIntegrateNonSeasonal()
	} else if hasSeasonalI {
		params.integrateSeasonal(dataForecastStationary)
		forecastMerged = params.getLastIntegrateSeasonal()
	} else if hasNonSeasonalI {
		params.integrateNonSeasonal(dataForecastStationary)
		forecastMerged = params.getLastIntegrateNonSeasonal()
	} else {
		forecastMerged = make([]float64, len(dataForecastStationary))
		// copy(dataForecastStationary, forecastMerged)
		copy(forecastMerged, dataForecastStationary)
	}

	return forecastMerged
}

func computeRMSE(left []float64, right []float64,
	leftIndexOffset, startIndex, endIndex int) float64 {

	len_left := len(left)
	len_right := len(right)
	if startIndex >= endIndex || startIndex < 0 || len_right < endIndex ||
		len_left+leftIndexOffset < 0 || len_left+leftIndexOffset < endIndex {
		log.Fatalf("invalid arguments: startIndex=%d, endIndex=%d, len_left=%d, len_right=%d, leftOffset=%d",
			startIndex, endIndex, len_left, len_right, leftIndexOffset)
	}
	square_sum := 0.0
	for i := startIndex; i < endIndex; i++ {
		dataerror := left[i+leftIndexOffset] - right[i]
		square_sum += dataerror * dataerror
	}
	return math.Sqrt(square_sum / float64(endIndex-startIndex))
}

func computeRMSEValidation(data []float64,
	testDataPercentage float64, params Config) float64 {

	testDataLength := int(float64(len(data)) * testDataPercentage)
	trainingDataEndIndex := len(data) - testDataLength

	result := estimateARIMA(params, data, trainingDataEndIndex, len(data))

	forecast := result.forecast(testDataLength).GetForecast()

	return computeRMSE(data, forecast, trainingDataEndIndex, 0, len(forecast))
}

func setSigma2AndPredicationInterval(params Config,
	forecastResult *Result, forecastSize int) float64 {

	coeffs_AR := params.getCurrentARCoefficients()
	coeffs_MA := params.getCurrentMACoefficients()
	return forecastResult.SetConfInterval(confidence_constant_95pct,
		getCumulativeSumOfCoeff(
			ARMAtoMA(coeffs_AR, coeffs_MA, forecastSize)))
}

func checkARIMADataLength(params Config, data []float64, startIndex, endIndex int) bool {
	result := true

	initialConditionSize := int(params.d + params.D*params.m)

	if len(data) < initialConditionSize || startIndex < initialConditionSize || endIndex <= startIndex {
		result = false
	}

	return result
}

/**
 * Hannan-Rissanen algorithm for estimating ARMA parameters
 */

func estimateARMA(data_orig []float64, params *Config,
	forecast_length, maxIteration int) {
	data := make([]float64, len(data_orig))
	total_length := len(data)
	// copy(data_orig, data)
	copy(data, data_orig)
	r := int(math.Max(float64(1+params.getDegreeP()), float64(1+params.getDegreeQ())))
	length := total_length - forecast_length
	size := length - r
	if length < (2 * r) {
		log.Fatalf("not enough data points: length= %d, r=", length, r)
	}

	// step 1: apply Yule-Walker method and estimate AR(r) model on input data
	errors := make([]float64, length)
	// yuleWalkerParams := applyYuleWalkerAndGetInitialErrors(data, r, length, errors)
	_ = applyYuleWalkerAndGetInitialErrors(data, r, length, errors)
	for j := 0; j < r; j++ {
		errors[j] = 0
	}

	// step 2: iterate Least-Square fitting until the parameters converge
	// instantiate Z-matrix
	matrix := make([][]float64, params.getNumParamsP()+params.getNumParamsQ())
	for i, _ := range matrix {
		matrix[i] = make([]float64, size)
	}

	bestRMSE := float64(-1) // initial value
	remainIteration := maxIteration
	var bestParams *mtx.InsightsVector
	for remainIteration >= 0 {
		estimatedParams := iterationStep(*params, data, errors, matrix, r,
			length,
			size)
		// originalParams := params.getParamsIntoVector()
		params.setParamsFromVector(estimatedParams)

		// forecast for validation data and compute RMSE
		forecasts := forecastARMA(*params, data, length, len(data))
		anotherRMSE := computeRMSE(data, forecasts, length, 0, forecast_length)
		// update errors
		train_forecasts := forecastARMA(*params, data, r, len(data))
		for j := 0; j < size; j++ {
			errors[j+r] = data[j+r] - train_forecasts[j]
		}
		if bestRMSE < 0 || anotherRMSE < bestRMSE {
			bestParams = estimatedParams
			bestRMSE = anotherRMSE
		}
		remainIteration--
	}
	params.setParamsFromVector(bestParams)
}

func applyYuleWalkerAndGetInitialErrors(data []float64, r, length int, errors []float64) []float64 {
	yuleWalker := Fit(data, r)
	bsYuleWalker := NewBackShift(r, true)
	bsYuleWalker.initializeParams(false)
	// return array from YuleWalker is an array of size r whose
	// 0-th index element is lag 1 coefficient etc
	// hence shifting lag index by one and copy over to BackShift operator
	for j := 0; j < r; j++ {
		bsYuleWalker.setParam(j+1, yuleWalker[j])
	}
	m := 0

	// populate error array
	for m < r {
		errors[m] = 0
		m++
	} // initial r-elements are set to zero
	for m < length {
		// from then on, initial estimate of error terms are
		// Z_t = X_t - \phi_1 X_{t-1} - \cdots - \phi_r X_{t-r}
		errors[m] = data[m] - bsYuleWalker.getLinearCombinationFrom(data, m)
		m++
	}
	return yuleWalker
}

func iterationStep(
	params Config,
	data []float64, errors []float64,
	matrix [][]float64, r, length, size int) *mtx.InsightsVector {

	rowIdx := 0
	// copy over shifted timeseries data into matrix
	offsetsAR := params.getOffsetsAR()
	for _, pIdx := range offsetsAR {
		// copy(data[r-pIdx:], matrix[rowIdx][:size])
		copy(matrix[rowIdx][:size], data[r-pIdx:])
		rowIdx++
	}
	// copy over shifted errors into matrix
	offsetsMA := params.getOffsetsMA()
	for _, qIdx := range offsetsMA {
		// copy(errors[r-qIdx:], matrix[rowIdx][:size])
		copy(matrix[rowIdx][:size], errors[r-qIdx:])
		rowIdx++
	}

	// instantiate matrix to perform least squares algorithm
	zt := mtx.NewInsightsMatrixWithData(matrix, false)

	// instantiate target vector
	vector := make([]float64, size)
	// copy(data[r:], vector[:size])
	copy(vector[:size], data[r:])
	x := mtx.NewInsightVectorWithData(vector, false)

	// obtain least squares solution
	ztx := zt.TimesVector(x)
	ztz := zt.ComputeAAT()
	estimatedVector := ztz.SolveSPDIntoVector(ztx, maxConditionNumber)

	return estimatedVector
}
