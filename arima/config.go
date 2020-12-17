package arima

import (
	"fmt"

	"github.com/DoOR-Team/timeseries_forecasting/arima/matrix"
	"github.com/DoOR-Team/timeseries_forecasting/arima/utils"
)

type Config struct {
	p, d, q, P, D, Q, m int

	// ARMA part
	opAR                 *BackShift
	opMA                 *BackShift
	dp, dq, np, nq       int
	initSeasonal         [][]float64
	diffSeasonal         [][]float64
	integrateSeasonal    [][]float64
	initNonSeasonal      [][]float64
	diffNonSeasonal      [][]float64
	integrateNonSeasonal [][]float64
	lagsAR               []int
	paramsAR             []float64
	lagsMA               []int
	paramsMA             []float64
	// I part
	mean float64
}

func NewConfig(p, d, q, P, D, Q, m int) Config {
	config := Config{
		p:                    p,
		d:                    d,
		q:                    q,
		P:                    P,
		D:                    D,
		Q:                    Q,
		m:                    m,
		opAR:                 nil,
		opMA:                 nil,
		dp:                   0,
		dq:                   0,
		np:                   0,
		nq:                   0,
		initSeasonal:         nil,
		diffSeasonal:         nil,
		integrateSeasonal:    nil,
		initNonSeasonal:      nil,
		diffNonSeasonal:      nil,
		integrateNonSeasonal: nil,
		lagsAR:               nil,
		paramsAR:             nil,
		lagsMA:               nil,
		paramsMA:             nil,
		mean:                 0,
	}

	config.opAR = config.getNewOperatorAR()
	config.opMA = config.getNewOperatorAR()

	config.opAR.initializeParams(false)
	config.opMA.initializeParams(false)
	config.dp = config.opAR.getDegree()
	config.dq = config.opMA.getDegree()
	config.np = config.opAR.numParams()
	config.nq = config.opMA.numParams()
	if D > 0 && m > 0 {
		config.initSeasonal = make([][]float64, D)
		for i, _ := range config.initSeasonal {
			config.initSeasonal[i] = make([]float64, m)
		}
	}

	if d > 0 {
		config.initNonSeasonal = make([][]float64, d)
		for i, _ := range config.initNonSeasonal {
			config.initNonSeasonal[i] = make([]float64, 1)
		}
	}

	if D > 0 && m > 0 {
		config.diffSeasonal = make([][]float64, D)
	}

	if d > 0 {
		config.diffNonSeasonal = make([][]float64, d)
	}

	if D > 0 && m > 0 {
		config.integrateSeasonal = make([][]float64, D)
	}

	if d > 0 {
		config.integrateNonSeasonal = make([][]float64, d)
	}

	return config
}

func (c Config) getDegreeP() int {
	return c.dp
}

func (c Config) getDegreeQ() int {
	return c.dq
}

func (c Config) forecastOnePointARMA(data []float64, errors []float64,
	index int) float64 {
	estimateAR := c.opAR.getLinearCombinationFrom(data, index)
	estimateMA := c.opMA.getLinearCombinationFrom(errors, index)
	forecastValue := estimateAR + estimateMA
	return forecastValue
}

func (c Config) getNumParamsP() int {
	return c.np
}

func (c Config) getNumParamsQ() int {
	return c.nq
}

func (c Config) getOffsetsAR() []int {
	return c.opAR.paramOffsets()
}
func (c Config) getOffsetsMA() []int {
	return c.opMA.paramOffsets()
}

func (c Config) getLastIntegrateSeasonal() []float64 {
	return c.integrateSeasonal[c.D-1]
}

func (c Config) getLastIntegrateNonSeasonal() []float64 {
	return c.integrateNonSeasonal[c.d-1]
}

func (c Config) getLastDifferenceSeasonal() []float64 {
	return c.diffSeasonal[c.D-1]
}

func (c Config) getLastDifferenceNonSeasonal() []float64 {
	return c.diffNonSeasonal[c.d-1]
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
		c.opAR.setParam(pIdx, paramVec.Get(index))
		index++
	}
	for _, qIdx := range offsetsMA {
		c.opMA.setParam(qIdx, paramVec.Get(index))
		index++
	}
}
func (c Config) getParamsIntoVector() matrix.InsightsVector {
	index := 0
	paramVec := matrix.NewInsightVector(c.np+c.nq, 0.0)
	offsetsAR := c.getOffsetsAR()
	offsetsMA := c.getOffsetsMA()
	for _, pIdx := range offsetsAR {
		paramVec.Set(index, c.opAR.getParam(pIdx))
		index++
	}
	for _, qIdx := range offsetsMA {
		paramVec.Set(index, c.opMA.getParam(qIdx))
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
	return c.opAR.getCoefficientsFlattened()
}

func (c Config) getCurrentMACoefficients() []float64 {
	return c.opMA.getCoefficientsFlattened()
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
		c.diffSeasonal[j] = next
		init := c.initSeasonal[j]
		utils.Differentiate(current, next, init, c.m)
		current = next
	}
}

func (c Config) differentiateNonSeasonal(data []float64) {
	current := data
	for j := 0; j < c.d; j++ {
		next := make([]float64, len(current)-1)
		c.diffNonSeasonal[j] = next
		init := c.initNonSeasonal[j]
		utils.Differentiate(current, next, init, 1)
		current = next
	}
}

func (c Config) getIntegrateSeasonal(data []float64) {
	current := data
	for j := 0; j < c.D; j++ {
		next := make([]float64, len(current)+c.m)
		c.integrateSeasonal[j] = next
		init := c.initSeasonal[j]
		utils.Integrate(current, next, init, c.m)
		current = next
	}
}

func (c Config) getIntegrateNonSeasonal(data []float64) {
	current := data
	for j := 0; j < c.d; j++ {
		next := make([]float64, len(current)+1)
		c.integrateNonSeasonal[j] = next
		init := c.initNonSeasonal[j]
		utils.Integrate(current, next, init, 1)
		current = next
	}
}
