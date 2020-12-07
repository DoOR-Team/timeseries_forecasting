package matrix

import "math"

type InsightsMatrix struct {
	_m     int
	_n     int
	_data  [][]float64
	_valid bool

	// Secondary
	_cholZero bool
	_cholPos  bool
	_cholNeg  bool
	_cholD    []float64
	_cholL    [][]float64
}

func NewInsightsMatrixWithData(data [][]float64, makeDeepCopy bool) *InsightsMatrix {
	matrix := &InsightsMatrix{
		_m:        -1,
		_n:        -1,
		_data:     nil,
		_valid:    false,
		_cholZero: false,
		_cholPos:  false,
		_cholNeg:  false,
		_cholD:    nil,
		_cholL:    nil,
	}

	if isValid2D(data) {
		matrix._valid = true
		matrix._m = len(data)
		matrix._n = len(data[0])
		if !makeDeepCopy {
			matrix._data = data
		} else {
			matrix._data = copy2DArray(data)
		}
	}
	return matrix
}

func isValid2D(matrix [][]float64) bool {
	result := true
	if matrix == nil || matrix[0] == nil || len(matrix[0]) == 0 {
		panic("[InsightsMatrix][constructor] null data given")
	} else {
		row := len(matrix)
		col := len(matrix[0])
		for i := 1; i < row; i++ {
			if matrix[i] == nil || len(matrix[i]) != col {
				result = false
			}
		}
	}
	return result
}

func copy2DArray(source [][]float64) [][]float64 {
	if source == nil {
		return nil
	} else if len(source) == 0 {
		return make([][]float64, 0)
	}

	row := len(source)
	target := make([][]float64, row)
	for i := 0; i < row; i++ {
		if source[i] == nil {
			target[i] = nil
		} else {
			rowLength := len(source[i])
			target[i] = make([]float64, rowLength)
			// copy(source[i], target[i])
			copy(target[i], source[i])
		}
	}
	return target
}

func (m *InsightsMatrix) GetNumberOfRows() int {
	return m._m
}

func (m *InsightsMatrix) GetNumberOfColumns() int {
	return m._n
}

func (m *InsightsMatrix) Get(i, j int) float64 {
	return m._data[i][j]
}

func (m *InsightsMatrix) Set(i, j int, val float64) {
	m._data[i][j] = val
}

func (m *InsightsMatrix) TimesVector(v *InsightsVector) *InsightsVector {
	if !m._valid || !v._valid || m._n != v._m {
		panic("[InsightsMatrix][timesVector] size mismatch")
	}
	data := make([]float64, m._m)
	var dotProduc float64
	for i := 0; i < m._m; i++ {
		rowVector := NewInsightVectorWithData(m._data[i], false)
		dotProduc = rowVector.Dot(v)
		data[i] = dotProduc
	}
	return NewInsightVectorWithData(data, false)
}

func (m *InsightsMatrix) computeCholeskyDecomposition(maxConditionNumber float64) bool {
	m._cholD = make([]float64, m._m)
	m._cholL = make([][]float64, m._m)

	for i, _ := range m._cholL {
		m._cholL[i] = make([]float64, m._n)
	}
	var i, j, k int
	var val float64
	var currentMax = -1.0
	// Backward marching method
	for j = 0; j < m._n; j++ {
		val = 0
		for k = 0; k < j; k++ {
			val += m._cholD[k] * m._cholL[j][k] * m._cholL[j][k]
		}
		diagTemp := m._data[j][j] - val
		diagSign := signum(diagTemp)
		switch diagSign {
		case 0: // singular diagonal value detected
			if maxConditionNumber < -0.5 { // no bound on maximum condition number
				m._cholZero = true
				m._cholL = nil
				m._cholD = nil
				return false
			} else {
				m._cholPos = true
			}
			break
		case 1:
			m._cholPos = true
			break
		case -1:
			m._cholNeg = true
			break
		}
		if maxConditionNumber > -0.5 {
			if currentMax <= 0.0 { // this is the first time
				if diagSign == 0 {
					diagTemp = 1.0
				}
			} else { // there was precedent
				if diagSign == 0 {
					diagTemp = math.Abs(currentMax / maxConditionNumber)
				} else {
					if math.Abs(diagTemp*maxConditionNumber) < currentMax {
						diagTemp = float64(diagSign) * math.Abs(currentMax/maxConditionNumber)
					}
				}
			}
		}
		m._cholD[j] = diagTemp
		if math.Abs(diagTemp) > currentMax {
			currentMax = math.Abs(diagTemp)
		}
		m._cholL[j][j] = 1
		for i = j + 1; i < m._m; i++ {
			val = 0
			for k = 0; k < j; k++ {
				val += m._cholD[k] * m._cholL[j][k] * m._cholL[i][k]
			}
			val = ((m._data[i][j]+m._data[j][i])/2 - val) / m._cholD[j]
			m._cholL[j][i] = val
			m._cholL[i][j] = val
		}
	}
	return true
}

func signum(temp float64) int {
	if temp == 0 {
		return 0
	} else if temp > 0 {
		return 1
	} else {
		return -1
	}
}

func (m *InsightsMatrix) SolveSPDIntoVector(b *InsightsVector, maxConditionNumber float64) *InsightsVector {
	if !m._valid || m._n != b._m {
		// invalid linear system
		panic(
			"[InsightsMatrix][solveSPDIntoVector] invalid linear system")
	}
	if m._cholL == nil {
		// computing Cholesky Decomposition
		m.computeCholeskyDecomposition(maxConditionNumber)
	}
	if m._cholZero {
		// singular matrix. returning null
		return nil
	}

	y := make([]float64, m._m)
	bt := make([]float64, m._n)
	var i, j int
	for i = 0; i < m._m; i++ {
		bt[i] = b._data[i]
	}
	var val float64
	for i = 0; i < m._m; i++ {
		val = 0
		for j = 0; j < i; j++ {
			val += m._cholL[i][j] * y[j]
		}
		y[i] = bt[i] - val
	}
	for i = m._m - 1; i >= 0; i-- {
		val = 0
		for j = i + 1; j < m._n; j++ {
			val += m._cholL[i][j] * bt[j]
		}
		bt[i] = y[i]/m._cholD[i] - val
	}
	return NewInsightVectorWithData(bt, false)
}

func (m *InsightsMatrix) ComputeAAT() *InsightsMatrix {
	if !m._valid {
		panic("[InsightsMatrix][computeAAT] invalid matrix")
	}
	data := make([][]float64, m._m)
	for i, _ := range data {
		data[i] = make([]float64, m._m)
	}

	for i := 0; i < m._m; i++ {
		rowI := m._data[i]
		for j := 0; j < m._m; j++ {
			rowJ := m._data[j]
			temp := 0.
			for k := 0; k < m._n; k++ {
				temp += rowI[k] * rowJ[k]
			}
			data[i][j] = temp
		}
	}
	return NewInsightsMatrixWithData(data, false)
}
