package arima

import (
	"fmt"

	"github.com/DoOR-Team/timeseries_forecasting/arima/matrix"
	"github.com/DoOR-Team/timeseries_forecasting/arima/utils"
)

type Config struct {
	p, d, q, P, D, Q, m int

	// ARMA part
	_opAR                   *BackShift
	_opMA                   *BackShift
	_dp, _dq, _np, _nq      int
	_init_seasonal          [][]float64
	_diff_seasonal          [][]float64
	_integrate_seasonal     [][]float64
	_init_non_seasonal      [][]float64
	_diff_non_seasonal      [][]float64
	_integrate_non_seasonal [][]float64
	lagsAR                  []int
	paramsAR                []float64
	lagsMA                  []int
	paramsMA                []float64
	// I part
	_mean float64
}

func NewConfig(p, d, q, P, D, Q, m int) Config {
	config := Config{
		p:                       p,
		d:                       d,
		q:                       q,
		P:                       P,
		D:                       D,
		Q:                       Q,
		m:                       m,
		_opAR:                   nil,
		_opMA:                   nil,
		_dp:                     0,
		_dq:                     0,
		_np:                     0,
		_nq:                     0,
		_init_seasonal:          nil,
		_diff_seasonal:          nil,
		_integrate_seasonal:     nil,
		_init_non_seasonal:      nil,
		_diff_non_seasonal:      nil,
		_integrate_non_seasonal: nil,
		lagsAR:                  nil,
		paramsAR:                nil,
		lagsMA:                  nil,
		paramsMA:                nil,
		_mean:                   0,
	}

	config._opAR = config.getNewOperatorAR()
	config._opMA = config.getNewOperatorAR()

	config._opAR.initializeParams(false)
	config._opMA.initializeParams(false)
	config._dp = config._opAR.getDegree()
	config._dq = config._opMA.getDegree()
	config._np = config._opAR.numParams()
	config._nq = config._opMA.numParams()
	if D > 0 && m > 0 {
		config._init_seasonal = make([][]float64, D)
		for i, _ := range config._init_seasonal {
			config._init_seasonal[i] = make([]float64, m)
		}
	}

	if d > 0 {
		config._init_non_seasonal = make([][]float64, d)
		for i, _ := range config._init_non_seasonal {
			config._init_non_seasonal[i] = make([]float64, 1)
		}
	}

	if D > 0 && m > 0 {
		config._diff_seasonal = make([][]float64, D)
	}

	if d > 0 {
		config._diff_non_seasonal = make([][]float64, d)
	}

	if D > 0 && m > 0 {
		config._integrate_seasonal = make([][]float64, D)
	}

	if d > 0 {
		config._integrate_non_seasonal = make([][]float64, d)
	}

	return config
}

func (c Config) getDegreeP() int {
	return c._dp
}

func (c Config) getDegreeQ() int {
	return c._dq
}

func (c Config) forecastOnePointARMA(data []float64, errors []float64,
	index int) float64 {
	estimateAR := c._opAR.getLinearCombinationFrom(data, index)
	estimateMA := c._opMA.getLinearCombinationFrom(errors, index)
	forecastValue := estimateAR + estimateMA
	return forecastValue
}

func (c Config) getNumParamsP() int {
	return c._np
}

func (c Config) getNumParamsQ() int {
	return c._nq
}

func (c Config) getOffsetsAR() []int {
	return c._opAR.paramOffsets()
}
func (c Config) getOffsetsMA() []int {
	return c._opMA.paramOffsets()
}

func (c Config) getLastIntegrateSeasonal() []float64 {
	return c._integrate_seasonal[c.D-1]
}

func (c Config) getLastIntegrateNonSeasonal() []float64 {
	return c._integrate_non_seasonal[c.d-1]
}

func (c Config) getLastDifferenceSeasonal() []float64 {
	return c._diff_seasonal[c.D-1]
}

func (c Config) getLastDifferenceNonSeasonal() []float64 {
	return c._diff_non_seasonal[c.d-1]
}

func (c Config) String() string {
	return fmt.Sprintf("ModelInterface ParamsInterface:"+
		", p= %d"+
		", d= %d"+
		", q= %d"+
		", P= %d"+
		", D= %d"+
		", Q= %d"+
		", m= %d", c.p, c.d, c.q, c.P, c.D, c.Q, c.m)
}

func (c Config) setParamsFromVector(paramVec *matrix.InsightsVector) {
	index := 0
	offsetsAR := c.getOffsetsAR()
	offsetsMA := c.getOffsetsMA()
	for _, pIdx := range offsetsAR {
		c._opAR.setParam(pIdx, paramVec.Get(index))
		index++
	}
	for _, qIdx := range offsetsMA {
		c._opMA.setParam(qIdx, paramVec.Get(index))
		index++
	}
}
func (c Config) getParamsIntoVector() matrix.InsightsVector {
	index := 0
	paramVec := matrix.NewInsightVector(c._np+c._nq, 0.0)
	offsetsAR := c.getOffsetsAR()
	offsetsMA := c.getOffsetsMA()
	for _, pIdx := range offsetsAR {
		paramVec.Set(index, c._opAR.getParam(pIdx))
		index++
	}
	for _, qIdx := range offsetsMA {
		paramVec.Set(index, c._opMA.getParam(qIdx))
		index++
	}
	return paramVec
}

func (c Config) getNewOperatorAR() *BackShift {
	return c.mergeSeasonalWithNonSeasonal(c.p, c.P, c.m)
}
func (c Config) getNewOperatorMA() *BackShift {
	return c.mergeSeasonalWithNonSeasonal(c.q, c.Q, c.m)
}

func (c Config) getCurrentARCoefficients() []float64 {
	return c._opAR.getCoefficientsFlattened()
}

func (c Config) getCurrentMACoefficients() []float64 {
	return c._opMA.getCoefficientsFlattened()
}

func (c Config) mergeSeasonalWithNonSeasonal(nonSeasonalLag, seasonalLag, seasonalStep int) *BackShift {
	nonSeasonal := NewBackShift(nonSeasonalLag, true)
	seasonal := NewBackShift(seasonalLag*seasonalStep, false)
	for s := 1; s <= seasonalLag; s++ {
		seasonal.setIndex(s*seasonalStep, true)
	}
	merged := seasonal.apply(nonSeasonal)
	return merged
}

// ================================
// Differentiation and Integration

func (c Config) differentiateSeasonal(data []float64) {
	current := data
	for j := 0; j < c.D; j++ {
		next := make([]float64, len(current)-c.m)
		c._diff_seasonal[j] = next
		init := c._init_seasonal[j]
		utils.Differentiate(current, next, init, c.m)
		current = next
	}
}

func (c Config) differentiateNonSeasonal(data []float64) {
	current := data
	for j := 0; j < c.d; j++ {
		next := make([]float64, len(current)-1)
		c._diff_non_seasonal[j] = next
		init := c._init_non_seasonal[j]
		utils.Differentiate(current, next, init, 1)
		current = next
	}
}

func (c Config) integrateSeasonal(data []float64) {
	current := data
	for j := 0; j < c.D; j++ {
		next := make([]float64, len(current)+c.m)
		c._integrate_seasonal[j] = next
		init := c._init_seasonal[j]
		utils.Integrate(current, next, init, c.m)
		current = next
	}
}

func (c Config) integrateNonSeasonal(data []float64) {
	current := data
	for j := 0; j < c.d; j++ {
		next := make([]float64, len(current)+1)
		c._integrate_non_seasonal[j] = next
		init := c._init_non_seasonal[j]
		utils.Integrate(current, next, init, 1)
		current = next
	}
}
