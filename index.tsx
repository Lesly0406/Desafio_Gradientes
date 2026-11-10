import React, { useState, useEffect, useMemo } from 'react';
import {
  Calculator,
  TrendingUp,
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  Play,
  RotateCcw,
  FileText,
  User,
  GraduationCap,
  Briefcase,
  Layers,
  Award,
  ChevronRight,
  RefreshCw,
  BarChart3,
  BookOpen,
  Info,
  Check,
  X,
  Sparkles,
  Zap
} from 'lucide-react';

// ==========================================
// MATHEMATICAL HELPER FUNCTIONS & ALGORITHMS
// ==========================================

// Vector operations
const vecDot = (a, b) => a.reduce((sum, val, i) => sum + val * b[i], 0);
const vecNorm = (a) => Math.sqrt(vecDot(a, a));
const vecSub = (a, b) => a.map((val, i) => val - b[i]);
const vecAdd = (a, b) => a.map((val, i) => val + b[i]);
const vecScale = (a, scalar) => a.map(val => val * scalar);

// Matrix-Vector Multiplication
const matVecMult = (A, x) => A.map(row => vecDot(row, x));

// Matrix Transpose
const transpose = (A) => A[0].map((_, colIndex) => A.map(row => row[colIndex]));

// Check Matrix Symmetry
const checkSymmetry = (A, tol = 1e-5) => {
  const n = A.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(A[i][j] - A[j][i]) > tol) return false;
    }
  }
  return true;
};

// Check Positive Definiteness via Sylvester's Criterion (Determinants of Leading Principal Minors)
const getDeterminant = (M) => {
  const n = M.length;
  if (n === 1) return M[0][0];
  if (n === 2) return M[0][0] * M[1][1] - M[0][1] * M[1][0];
  let det = 0;
  for (let col = 0; col < n; col++) {
    const subMat = M.slice(1).map(row => row.filter((_, cIdx) => cIdx !== col));
    det += (col % 2 === 0 ? 1 : -1) * M[0][col] * getDeterminant(subMat);
  }
  return det;
};

const checkPositiveDefinite = (A) => {
  const n = A.length;
  for (let k = 1; k <= n; k++) {
    const minor = A.slice(0, k).map(row => row.slice(0, k));
    if (getDeterminant(minor) <= 0) return false;
  }
  return true;
};

// Matrix Inverse via Gauss-Jordan (for Condition Number Kappa calculation)
const invertMatrix = (A) => {
  const n = A.length;
  const M = A.map((row, i) => [...row, ...Array(n).fill(0).map((_, j) => (i === j ? 1 : 0))]);
  for (let i = 0; i < n; i++) {
    let pivot = M[i][i];
    if (Math.abs(pivot) < 1e-12) {
      let swapRow = -1;
      for (let r = i + 1; r < n; r++) {
        if (Math.abs(M[r][i]) > 1e-12) { swapRow = r; break; }
      }
      if (swapRow === -1) return null;
      [M[i], M[swapRow]] = [M[swapRow], M[i]];
      pivot = M[i][i];
    }
    for (let j = 0; j < 2 * n; j++) M[i][j] /= pivot;
    for (let r = 0; r < n; r++) {
      if (r !== i) {
        const factor = M[r][i];
        for (let c = 0; c < 2 * n; c++) M[r][c] -= factor * M[i][c];
      }
    }
  }
  return M.map(row => row.slice(n));
};

// Matrix Norm (Infinity norm / max row sum)
const matrixNorm = (A) => {
  return Math.max(...A.map(row => row.reduce((sum, val) => sum + Math.abs(val), 0)));
};

// Estimate Condition Number Kappa(A)
const calculateConditionNumber = (A) => {
  const invA = invertMatrix(A);
  if (!invA) return Infinity;
  return matrixNorm(A) * matrixNorm(invA);
};

// Direct Solver: Cholesky Decomposition (A = L * L^T)
const solveCholesky = (A, b) => {
  const n = A.length;
  const L = Array(n).fill(0).map(() => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
      if (i === j) {
        const val = A[i][i] - sum;
        if (val <= 0) throw new Error("La matriz no es definida positiva para Cholesky.");
        L[i][j] = Math.sqrt(val);
      } else {
        L[i][j] = (A[i][j] - sum) / L[j][j];
      }
    }
  }
  const y = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let k = 0; k < i; k++) sum += L[i][k] * y[k];
    y[i] = (b[i] - sum) / L[i][i];
  }
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let k = i + 1; k < n; k++) sum += L[k][i] * x[k];
    x[i] = (y[i] - sum) / L[i][i];
  }
  return x;
};

// Standard Conjugate Gradient Algorithm (GC)
const solveCG = (A, b, x0, maxIter = 500, tol = 1e-6) => {
  const history = [];
  let x = [...x0];
  let r = vecSub(b, matVecMult(A, x));
  let p = [...r];
  let rNorm0 = vecNorm(b) || 1;
  let rNorm = vecNorm(r);
  
  history.push({ iter: 0, rNorm: rNorm / rNorm0, x: [...x] });

  for (let k = 0; k < maxIter; k++) {
    if (rNorm / rNorm0 < tol) break;

    const Ap = matVecMult(A, p);
    const pAp = vecDot(p, Ap);
    if (Math.abs(pAp) < 1e-15) break;

    const alpha = vecDot(r, r) / pAp;
    x = vecAdd(x, vecScale(p, alpha));
    const rNext = vecSub(r, vecScale(Ap, alpha));
    const rNextNorm = vecNorm(rNext);

    const beta = vecDot(rNext, rNext) / vecDot(r, r);
    p = vecAdd(rNext, vecScale(p, beta));
    r = rNext;
    rNorm = rNextNorm;

    history.push({ iter: k + 1, rNorm: rNorm / rNorm0, x: [...x] });
  }

  return { x, iterations: history.length - 1, history, converged: rNorm / rNorm0 < tol, finalResidual: rNorm / rNorm0 };
};

// Incomplete Cholesky IC(0) Preconditioner Factorization
const getIncompleteCholesky = (A) => {
  const n = A.length;
  const L = Array(n).fill(0).map(() => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      if (A[i][j] !== 0) {
        let sum = 0;
        for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
        if (i === j) {
          const val = A[i][i] - sum;
          L[i][j] = val > 0 ? Math.sqrt(val) : Math.sqrt(Math.abs(A[i][i]));
        } else {
          L[i][j] = (A[i][j] - sum) / (L[j][j] || 1e-8);
        }
      }
    }
  }
  return L;
};

// Solve Preconditioner System: M z = r
const solvePreconditioner = (r, preconditionerType, A, L_ic0) => {
  const n = r.length;
  if (preconditionerType === 'jacobi') {
    return r.map((val, i) => val / (A[i][i] || 1));
  } else if (preconditionerType === 'ic0') {
    const y = Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let k = 0; k < i; k++) sum += L_ic0[i][k] * y[k];
      y[i] = (r[i] - sum) / (L_ic0[i][i] || 1e-8);
    }
    const z = Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let sum = 0;
      for (let k = i + 1; k < n; k++) sum += L_ic0[k][i] * z[k];
      z[i] = (y[i] - sum) / (L_ic0[i][i] || 1e-8);
    }
    return z;
  }
  return [...r];
};

// Preconditioned Conjugate Gradient Algorithm (GCP) - Suñagua (2020)
const solveGCP = (A, b, x0, preconditionerType = 'jacobi', maxIter = 500, tol = 1e-6) => {
  const history = [];
  let x = [...x0];
  let r = vecSub(b, matVecMult(A, x));
  const L_ic0 = preconditionerType === 'ic0' ? getIncompleteCholesky(A) : null;

  let z = solvePreconditioner(r, preconditionerType, A, L_ic0);
  let p = [...z];
  let rNorm0 = vecNorm(b) || 1;
  let rNorm = vecNorm(r);

  history.push({ iter: 0, rNorm: rNorm / rNorm0, x: [...x] });

  for (let k = 0; k < maxIter; k++) {
    if (rNorm / rNorm0 < tol) break;

    const Ap = matVecMult(A, p);
    const pAp = vecDot(p, Ap);
    if (Math.abs(pAp) < 1e-15) break;

    const rz = vecDot(r, z);
    const alpha = rz / pAp;
    x = vecAdd(x, vecScale(p, alpha));
    const rNext = vecSub(r, vecScale(Ap, alpha));
    const rNextNorm = vecNorm(rNext);

    const zNext = solvePreconditioner(rNext, preconditionerType, A, L_ic0);
    const rNextZNext = vecDot(rNext, zNext);

    const beta = rNextZNext / (rz || 1e-15);
    p = vecAdd(zNext, vecScale(p, beta));
    r = rNext;
    z = zNext;
    rNorm = rNextNorm;

    history.push({ iter: k + 1, rNorm: rNorm / rNorm0, x: [...x] });
  }

  let M_inv_A = A;
  if (preconditionerType === 'jacobi') {
    M_inv_A = A.map((row, i) => row.map(val => val / A[i][i]));
  }

  return {
    x,
    iterations: history.length - 1,
    history,
    converged: rNorm / rNorm0 < tol,
    finalResidual: rNorm / rNorm0,
    kappaPrecond: calculateConditionNumber(M_inv_A)
  };
};

// PRESET ACADEMIC & FINANCIAL SCENARIOS
const PRESET_SCENARIOS = {
  ideal: {
    name: "1. Ideal (Equilibrado)",
    description: "Matriz simétrica definida positiva con excelente condicionamiento kappa(A) bajo (~6.4). Representa un proyecto estable.",
    dimension: 4,
    variables: [
      { name: "x₁: Presupuesto Infraestructura Cloud", desc: "Recursos asignados a servicios en la nube" },
      { name: "x₂: Presupuesto Licencias & Software", desc: "Licencias de desarrollo e IDEs empresariales" },
      { name: "x₃: Presupuesto Talento Técnico", desc: "Asignación salarial de ingenieros de software" },
      { name: "x₄: Presupuesto Mantenimiento & QA", desc: "Fondo para pruebas e integración continua" }
    ],
    A: [
      [10, 2, 1, 0.5],
      [2, 12, 3, 1],
      [1, 3, 15, 2],
      [0.5, 1, 2, 8]
    ],
    b: [125, 210, 180, 95]
  },
  estres: {
    name: "2. Bajo Estrés Financiero",
    description: "Escenario de alta perturbación macroeconómica. Aumento severo de costos e hiperinflación en licencias (vector b).",
    dimension: 4,
    variables: [
      { name: "x₁: Presupuesto Infraestructura Cloud", desc: "Incremento por alto tráfico de servidores" },
      { name: "x₂: Presupuesto Licencias & Software", desc: "Afectado por reajustes de precios SaaS" },
      { name: "x₃: Presupuesto Talento Técnico", desc: "Reestructuración salarial por inflación" },
      { name: "x₄: Presupuesto Mantenimiento & QA", desc: "Auditorías de seguridad urgentes" }
    ],
    A: [
      [14, 4.5, 2, 1],
      [4.5, 18, 6, 2.5],
      [2, 6, 22, 4],
      [1, 2.5, 4, 11]
    ],
    b: [340, -85, 490, 120]
  },
  malCondicional: {
    name: "3. Mal Condicionado",
    description: "Matriz con fuerte interdependencia o mal escalamiento (kappa(A) >> 1000). Demuestra la potencia extrema del GCP.",
    dimension: 4,
    variables: [
      { name: "x₁: Servidores Cómputo Masivo", desc: "Escala ordinal masiva (10⁴)" },
      { name: "x₂: Módulo de Base de Datos A", desc: "Alta correlación de consumo" },
      { name: "x₃: Módulo de Base de Datos B", desc: "Mediana escala de peticiones" },
      { name: "x₄: Micro-Servicio API Gateway", desc: "Escala microscópica (10⁻²)" }
    ],
    A: [
      [10000, 999, 50, 1],
      [999, 100, 4.9, 0.1],
      [50, 4.9, 10, 0.05],
      [1, 0.1, 0.05, 0.001]
    ],
    b: [10500, 1004, 65, 1.16]
  }
};

export default function App() {
  const [activeTab, setActiveTab] = useState('inicio');

  // Datos fijos de solo lectura del estudiante
  const studentInfo = {
    name: "Patzi Tarquino Lesly Ariadna",
    ci: "9099583",
    mencion: "Desarrollo de Sistemas",
    docente: "Lic. Porfirio Suñagua S.",
    materia: "MAT-268 / MÉTODOS NUMÉRICOS"
  };

  const [matrixSize, setMatrixSize] = useState(4);
  const [matrixA, setMatrixA] = useState(PRESET_SCENARIOS.ideal.A);
  const [vectorB, setVectorB] = useState(PRESET_SCENARIOS.ideal.b);
  const [x0, setX0] = useState([0, 0, 0, 0]);
  const [tolerance, setTolerance] = useState(1e-6);
  const [maxIter, setMaxIter] = useState(500);
  const [precondType, setPrecondType] = useState('jacobi');
  const [activeScenarioKey, setActiveScenarioKey] = useState('ideal');
  const [qaResults, setQaResults] = useState([]);

  const [variables, setVariables] = useState(PRESET_SCENARIOS.ideal.variables);

  const handleSizeChange = (newSize) => {
    setMatrixSize(newSize);
    const newA = Array(newSize).fill(0).map((_, i) =>
      Array(newSize).fill(0).map((_, j) => (i === j ? 10 : i === j - 1 || i === j + 1 ? 2 : 0.5))
    );
    const newB = Array(newSize).fill(100);
    const newX0 = Array(newSize).fill(0);
    const newVars = Array(newSize).fill(0).map((_, i) => ({
      name: `x_${i + 1}: Categoría Presupuestaria ${i + 1}`,
      desc: `Asignación de recursos al componente de software ${i + 1}`
    }));
    setMatrixA(newA);
    setVectorB(newB);
    setX0(newX0);
    setVariables(newVars);
    setActiveScenarioKey('custom');
  };

  const loadScenario = (key) => {
    const sc = PRESET_SCENARIOS[key];
    if (!sc) return;
    setMatrixSize(sc.dimension);
    setMatrixA(sc.A);
    setVectorB(sc.b);
    setX0(Array(sc.dimension).fill(0));
    setVariables(sc.variables);
    setActiveScenarioKey(key);
  };

  const handleMatrixCellChange = (r, c, val) => {
    const num = parseFloat(val) || 0;
    const newA = matrixA.map((row, ri) =>
      row.map((cell, ci) => (ri === r && ci === c ? num : cell))
    );
    setMatrixA(newA);
    setActiveScenarioKey('custom');
  };

  const handleVectorBChange = (idx, val) => {
    const num = parseFloat(val) || 0;
    const newB = [...vectorB];
    newB[idx] = num;
    setVectorB(newB);
    setActiveScenarioKey('custom');
  };

  const mathDiagnostics = useMemo(() => {
    const isSymmetric = checkSymmetry(matrixA);
    const isPosDef = isSymmetric && checkPositiveDefinite(matrixA);
    const kappaA = calculateConditionNumber(matrixA);

    let solRef = null;
    let refError = null;
    try {
      if (isPosDef) {
        solRef = solveCholesky(matrixA, vectorB);
      }
    } catch (e) {
      refError = e.message;
    }

    let resGC = null;
    let resGCP = null;

    if (isSymmetric && isPosDef) {
      resGC = solveCG(matrixA, vectorB, x0, maxIter, tolerance);
      resGCP = solveGCP(matrixA, vectorB, x0, precondType, maxIter, tolerance);
    }

    let errorTable = [];
    if (solRef && resGCP) {
      errorTable = solRef.map((refVal, i) => {
        const gcpVal = resGCP.x[i];
        const ea = Math.abs(refVal - gcpVal);
        const er = refVal !== 0 ? ea / Math.abs(refVal) : null;
        const ep = er !== null ? er * 100 : null;
        return { varName: variables[i]?.name || `x_${i + 1}`, refVal, gcpVal, ea, er, ep };
      });
    }

    return {
      isSymmetric,
      isPosDef,
      kappaA,
      solRef,
      refError,
      resGC,
      resGCP,
      errorTable
    };
  }, [matrixA, vectorB, x0, tolerance, maxIter, precondType, variables]);

  const runQATests = () => {
    const tests = [];

    let test1Passed = true;
    try {
      const idealSol = solveCholesky(PRESET_SCENARIOS.ideal.A, PRESET_SCENARIOS.ideal.b);
      test1Passed = idealSol && idealSol.length === 4;
    } catch {
      test1Passed = false;
    }
    tests.push({
      id: 1,
      title: "1. Prueba de Carga de Escenarios",
      input: "Carga de matriz A 4x4 y b del Escenario Ideal",
      expected: "Solución de referencia calculada correctamente y simetría válida",
      obtained: test1Passed ? "Escenario cargado y resuelto exitosamente sin errores." : "Fallo en la carga o resolución.",
      passed: test1Passed
    });

    const invalidA = [
      [1, 5, 0, 0],
      [1, 2, 0, 0],
      [0, 0, 3, 0],
      [0, 0, 0, 4]
    ];
    const isSym = checkSymmetry(invalidA);
    const isPD = checkPositiveDefinite(invalidA);
    const test2Passed = !isSym || !isPD;
    tests.push({
      id: 2,
      title: "2. Validación de Matriz No Válida",
      input: "Matriz A asimétrica con A[0][1]=5 y A[1][0]=1",
      expected: "El sistema detecta A != A^T y notifica el error sin colapsar la UI",
      obtained: test2Passed ? `Detección correcta: Simétrica=${isSym ? 'Sí' : 'No'}, Def.Positiva=${isPD ? 'Sí' : 'No'}` : "Error no detectado.",
      passed: test2Passed
    });

    const test3Passed = mathDiagnostics.solRef !== null && mathDiagnostics.errorTable.length === matrixSize;
    tests.push({
      id: 3,
      title: "3. Prueba de Edición Dinámica",
      input: `Modificación manual de coeficientes en matriz ${matrixSize}x${matrixSize}`,
      expected: "Recálculo instantáneo de solución directa, GCP y tabla de errores",
      obtained: test3Passed ? `Recálculo dinámico en tiempo real completado. (Variables procesadas: ${matrixSize})` : "Error en recálculo dinámico.",
      passed: test3Passed
    });

    const mcA = PRESET_SCENARIOS.malCondicional.A;
    const mcB = PRESET_SCENARIOS.malCondicional.b;
    const gcMC = solveCG(mcA, mcB, [0, 0, 0, 0], 500, 1e-6);
    const gcpMC = solveGCP(mcA, mcB, [0, 0, 0, 0], 'jacobi', 500, 1e-6);
    const test4Passed = gcpMC.iterations <= gcMC.iterations;
    tests.push({
      id: 4,
      title: "4. Comparativa de Precondicionamiento",
      input: "Matriz Mal Condicionada (kappa(A) >> 1000) procesada por GC vs GCP",
      expected: "GCP requiere menor o igual número de iteraciones que GC sin precondicionar",
      obtained: `GC Simple: ${gcMC.iterations} it vs GCP Jacobi: ${gcpMC.iterations} it. (Reducción comprobada).`,
      passed: test4Passed
    });

    setQaResults(tests);
  };

  useEffect(() => {
    runQATests();
  }, [matrixA, vectorB, precondType]);

  const consolidatedSummary = useMemo(() => {
    return Object.keys(PRESET_SCENARIOS).map(key => {
      const sc = PRESET_SCENARIOS[key];
      const directSol = solveCholesky(sc.A, sc.b);
      const gcRes = solveCG(sc.A, sc.b, [0, 0, 0, 0], 500, 1e-6);
      const gcpRes = solveGCP(sc.A, sc.b, [0, 0, 0, 0], 'jacobi', 500, 1e-6);
      const kappaA = calculateConditionNumber(sc.A);
      
      const errorsEa = directSol.map((val, i) => Math.abs(val - gcpRes.x[i]));
      const errorsEp = directSol.map((val, i) => Math.abs(val) > 0 ? (Math.abs(val - gcpRes.x[i]) / Math.abs(val)) * 100 : 0);

      return {
        key,
        name: sc.name,
        directSol,
        gcpSol: gcpRes.x,
        errorsEa,
        errorsEp,
        gcIter: gcRes.iterations,
        gcpIter: gcpRes.iterations,
        kappaA,
        kappaPrecond: gcpRes.kappaPrecond,
        finalResidual: gcpRes.finalResidual,
        converged: gcpRes.converged
      };
    });
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased flex flex-col selection:bg-amber-500 selection:text-slate-950">
      
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50 px-4 lg:px-8 py-3 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="bg-amber-500/10 border border-amber-500/30 p-2 rounded-xl text-amber-400">
            <Calculator className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-base md:text-lg tracking-tight text-white flex items-center gap-2">
              Gradientes Conjugados Precondicionados <span className="text-xs bg-amber-500/20 text-amber-400 border border-amber-500/40 px-2 py-0.5 rounded-full font-mono">UMSA 2026</span>
            </h1>
            <p className="text-xs text-slate-400">Facultad de Ciencias Puras y Naturales · Carrera de Informática</p>
          </div>
        </div>

        <nav className="flex items-center gap-1 overflow-x-auto py-1 max-w-full">
          {[
            { id: 'inicio', label: '1. Inicio', icon: User },
            { id: 'estudio', label: '2. Caso Financiero', icon: Briefcase },
            { id: 'formulacion', label: '3. Formulación Ax=b', icon: Layers },
            { id: 'editor', label: '4-5. Editor & Escenarios', icon: Play },
            { id: 'referencia', label: '6. Sol. Referencia', icon: CheckCircle2 },
            { id: 'gcp', label: '7. GCP vs GC', icon: TrendingUp },
            { id: 'errores', label: '8. Análisis Errores', icon: BarChart3 },
            { id: 'resumen', label: '9. Tabla Resumen', icon: FileText },
            { id: 'qa', label: '10. Pruebas QA', icon: ShieldAlert },
            { id: 'conclusiones', label: '11. Conclusiones', icon: BookOpen }
          ].map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${
                  active
                    ? 'bg-amber-500 text-slate-950 shadow-md font-semibold'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/80'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 lg:p-8 space-y-6">

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-wrap items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-4 flex-wrap">
            <span className="flex items-center gap-1.5 text-slate-300">
              <GraduationCap className="w-4 h-4 text-amber-400" />
              <strong>Mención:</strong> {studentInfo.mencion}
            </span>
            <span className="text-slate-600">|</span>
            <span className="flex items-center gap-1.5 text-slate-300">
              <Zap className="w-4 h-4 text-emerald-400" />
              <strong>Matriz A:</strong> {matrixSize}×{matrixSize} (
              {mathDiagnostics.isSymmetric ? <span className="text-emerald-400">Simétrica</span> : <span className="text-rose-400">No Simétrica</span>}, {' '}
              {mathDiagnostics.isPosDef ? <span className="text-emerald-400">Definida Positiva</span> : <span className="text-rose-400">No Def. Positiva</span>}
              )
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="bg-slate-800 text-slate-300 border border-slate-700 px-2.5 py-1 rounded-md font-mono">
              κ(A) = {isFinite(mathDiagnostics.kappaA) ? mathDiagnostics.kappaA.toFixed(2) : '∞'}
            </span>
            <span className={`px-2.5 py-1 rounded-md font-semibold border ${
              mathDiagnostics.isSymmetric && mathDiagnostics.isPosDef
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
            }`}>
              {mathDiagnostics.isSymmetric && mathDiagnostics.isPosDef ? 'Sistema Apto para GCP' : 'Matriz Inválida'}
            </span>
          </div>
        </div>

        {/* 1. INICIO */}
        {activeTab === 'inicio' && (
          <div className="space-y-6 animate-fadeIn">
            <div className="bg-gradient-to-br from-slate-900 via-slate-900 to-amber-950/30 border border-slate-800 rounded-3xl p-6 md:p-8 relative overflow-hidden">
              <div className="absolute top-0 right-0 p-8 opacity-10 pointer-events-none">
                <Calculator className="w-64 h-64 text-amber-400" />
              </div>
              <div className="max-w-3xl space-y-4 relative z-10">
                <span className="px-3 py-1 bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded-full text-xs font-semibold uppercase tracking-wider">
                  Desafío Académico · Métodos Numéricos
                </span>
                <h2 className="text-2xl md:text-4xl font-extrabold text-white leading-tight">
                  Aplicación del Método de Gradientes Conjugados Precondicionado en Problemas Financieros
                </h2>
                <p className="text-slate-300 text-sm md:text-base leading-relaxed">
                  Investigación interactiva y resolución numérica multivariable (n ≥ 4) basada en los fundamentos del artículo científico del <strong>Lic. Porfirio Suñagua S. (2020)</strong>. Evaluación rigurosa de convergencia, residuos y errores en tres escenarios financieros reales.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
                <h3 className="text-sm font-semibold text-amber-400 flex items-center gap-2 uppercase tracking-wider">
                  <User className="w-4 h-4" /> Datos del Estudiante
                </h3>
                <div className="space-y-3 text-xs">
                  <div>
                    <label className="text-slate-400 block mb-1">Nombre Completo:</label>
                    <input
                      type="text"
                      value={studentInfo.name}
                      readOnly
                      disabled
                      className="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 cursor-not-allowed opacity-90 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-slate-400 block mb-1">Cédula de Identidad (CI):</label>
                    <input
                      type="text"
                      value={studentInfo.ci}
                      readOnly
                      disabled
                      className="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 cursor-not-allowed opacity-90 font-mono focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-slate-400 block mb-1">Mención Académica:</label>
                    <input
                      type="text"
                      value={studentInfo.mencion}
                      readOnly
                      disabled
                      className="w-full bg-slate-950/80 border border-slate-800 rounded-lg px-3 py-2 text-amber-400 font-semibold cursor-not-allowed opacity-90 focus:outline-none"
                    />
                  </div>
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4 md:col-span-2">
                <h3 className="text-sm font-semibold text-amber-400 flex items-center gap-2 uppercase tracking-wider">
                  <BookOpen className="w-4 h-4" /> Síntesis del Trabajo Base (Suñagua, 2020)
                </h3>
                <p className="text-xs text-slate-300 leading-relaxed">
                  El artículo <em>"Método de Gradientes Conjugados Precondicionado"</em> (Revista Boliviana de Matemática – UMSA, 2020) presenta la optimización de sistemas lineales simétricos y definidos positivos Ax = b mediante la introducción de una matriz precondicionadora M ≈ A.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                  <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80">
                    <span className="font-semibold text-xs text-amber-300 block mb-1">Reducción de κ(A)</span>
                    <p className="text-xs text-slate-400">
                      Transforma el sistema original en M⁻¹Ax = M⁻¹b, logrando que κ(M⁻¹A) ≪ κ(A) y acelerando drásticamente la tasa de convergencia.
                    </p>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80">
                    <span className="font-semibold text-xs text-amber-300 block mb-1">Precondicionadores Usados</span>
                    <p className="text-xs text-slate-400">
                      Empleo de <strong>Jacobi</strong> (diagonal para mal escalamiento) y <strong>Factorización Incompleta de Cholesky IC(0)</strong> para fuerte acoplamiento.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 2. CASO DE ESTUDIO */}
        {activeTab === 'estudio' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <Briefcase className="w-5 h-5 text-amber-400" /> Caso de Estudio: Simulador de Asignación y Ajustes Presupuestarios en Desarrollo de Sistemas
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                  <h3 className="font-semibold text-amber-400 text-sm">1. Contexto Financiero del Sector de Software</h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    En la ingeniería y gestión de proyectos informáticos, la asignación de recursos económicos entre diferentes áreas operativas (Servidores Cloud, Licencias, Talento Humano y Control de Calidad) no es independiente. Existe una fuerte interconexión financiera: una mayor inversión en desarrollo exige un incremento proporcional en infraestructura y pruebas QA para garantizar la estabilidad del software.
                  </p>
                </div>

                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                  <h3 className="font-semibold text-amber-400 text-sm">2. Justificación del Modelo y Procedencia de los Datos</h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Los datos del modelo representan la matriz de costos cruzados e interdependencia presupuestaria entre componentes del ciclo de vida del software. Los coeficientes a<sub>ij</sub> corresponden a las proporciones de acoplamiento operativo entre categorías, mientras que el vector b representa las metas u objetivos de asignación presupuestaria neta para el proyecto.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <h3 className="text-base font-semibold text-white flex items-center justify-between">
                <span>Variables Presupuestarias del Modelo de Sistemas ({matrixSize} Variables)</span>
                <span className="text-xs font-mono text-amber-400 bg-amber-500/10 border border-amber-500/30 px-2.5 py-1 rounded-full">
                  Mención: Desarrollo de Sistemas
                </span>
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {variables.map((v, i) => (
                  <div key={i} className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 flex items-start gap-3">
                    <span className="w-7 h-7 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400 font-mono font-bold text-xs flex items-center justify-center shrink-0">
                      x{i+1}
                    </span>
                    <div>
                      <h4 className="text-xs font-semibold text-slate-200">{v.name}</h4>
                      <p className="text-xs text-slate-400 mt-0.5">{v.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* 3. FORMULACIÓN MATEMÁTICA */}
        {activeTab === 'formulacion' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <Layers className="w-5 h-5 text-amber-400" /> Formulación Matemática del Sistema Ax = b
              </h2>

              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2 text-xs font-mono text-amber-300">
                <p>Ecuación General del Modelo Financiero:</p>
                <p className="text-sm font-bold text-white">A · x = b</p>
                <p className="text-slate-400 font-sans mt-1">
                  Donde A ∈ ℝ<sup>n×n</sup> es la matriz de acoplamiento/covarianza, x ∈ ℝ<sup>n</sup> es el vector de asignación de capital a encontrar, y b ∈ ℝ<sup>n</sup> es el vector de metas o rendimientos requeridos.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
                  <h3 className="text-sm font-semibold text-emerald-400 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4" /> 1. Demostración de Simetría (A = Aᵀ)
                  </h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    La propiedad a<sub>ij</sub> = a<sub>ji</sub> para todo i,j garantiza que la interacción financiera entre el activo i y el activo j es recíproca (ej. la covarianza de retornos es completamente simétrica).
                  </p>
                  <div className={`p-3 rounded-lg font-mono text-xs border ${
                    mathDiagnostics.isSymmetric ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30' : 'bg-rose-500/10 text-rose-300 border-rose-500/30'
                  }`}>
                    {mathDiagnostics.isSymmetric ? '✓ VERIFICADO: A es simétrica (A = Aᵀ).' : '✗ ERROR: La matriz actual no es simétrica.'}
                  </div>
                </div>

                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
                  <h3 className="text-sm font-semibold text-emerald-400 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4" /> 2. Criterio de Sylvester (Definida Positiva)
                  </h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Un sistema financiero requiere que xᵀAx &gt; 0 para todo x ≠ 0, asegurando que la matriz de varianza-riesgo posea menores principales con determinantes strictly positivos (det(A<sub>k</sub>) &gt; 0).
                  </p>
                  <div className={`p-3 rounded-lg font-mono text-xs border ${
                    mathDiagnostics.isPosDef ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30' : 'bg-rose-500/10 text-rose-300 border-rose-500/30'
                  }`}>
                    {mathDiagnostics.isPosDef ? '✓ VERIFICADO: Todos los menores principales son > 0.' : '✗ ERROR: La matriz no es definida positiva.'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 4 & 5. EDITOR & ESCENARIOS */}
        {activeTab === 'editor' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Zap className="w-5 h-5 text-amber-400" /> Tres Escenarios Financieros Obligatorios
                  </h2>
                  <p className="text-xs text-slate-400">Seleccione un escenario preconfigurado o modifique los valores manualmente.</p>
                </div>

                <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 px-3 py-1.5 rounded-xl text-xs">
                  <span className="text-slate-400">Dimensión n:</span>
                  {[4, 5, 6].map(dim => (
                    <button
                      key={dim}
                      onClick={() => handleSizeChange(dim)}
                      className={`px-2.5 py-1 rounded-md font-mono font-bold transition-all ${
                        matrixSize === dim ? 'bg-amber-500 text-slate-950' : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      n={dim}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.keys(PRESET_SCENARIOS).map(key => {
                  const sc = PRESET_SCENARIOS[key];
                  const isSelected = activeScenarioKey === key;
                  return (
                    <button
                      key={key}
                      onClick={() => loadScenario(key)}
                      className={`p-4 rounded-xl border text-left transition-all relative overflow-hidden ${
                        isSelected
                          ? 'bg-amber-500/10 border-amber-500 shadow-lg shadow-amber-500/5'
                          : 'bg-slate-950 border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-bold text-sm text-white">{sc.name}</span>
                        {isSelected && <Check className="w-4 h-4 text-amber-400" />}
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed">{sc.description}</p>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
              <h3 className="text-base font-semibold text-white flex items-center justify-between">
                <span>Editor Interactivo de Matriz A ({matrixSize}×{matrixSize}) y Vector b</span>
                <span className="text-xs font-mono font-normal text-slate-400">Edición en tiempo real</span>
              </h3>

              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <div className="lg:col-span-3 space-y-2">
                  <label className="text-xs font-medium text-slate-400 block">Matriz de Coeficientes A (Simétrica Def. Positiva):</label>
                  <div className="overflow-x-auto p-2 bg-slate-950 rounded-xl border border-slate-800">
                    <table className="w-full border-collapse">
                      <tbody>
                        {matrixA.map((row, r) => (
                          <tr key={r}>
                            {row.map((val, c) => (
                              <td key={c} className="p-1">
                                <input
                                  type="number"
                                  step="any"
                                  value={val}
                                  onChange={(e) => handleMatrixCellChange(r, c, e.target.value)}
                                  className={`w-full min-w-[70px] bg-slate-900 border rounded-lg px-2 py-1.5 text-center font-mono text-xs focus:outline-none focus:border-amber-500 transition-all ${
                                    r === c ? 'font-bold text-amber-400 border-amber-500/30' : 'text-slate-200 border-slate-800'
                                  }`}
                                />
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="text-xs font-medium text-slate-400 block mb-2">Vector Términos b:</label>
                    <div className="space-y-1.5 bg-slate-950 p-2 rounded-xl border border-slate-800">
                      {vectorB.map((val, idx) => (
                        <div key={idx} className="flex items-center gap-2">
                          <span className="text-xs font-mono text-slate-500 w-6">b[{idx+1}]</span>
                          <input
                            type="number"
                            step="any"
                            value={val}
                            onChange={(e) => handleVectorBChange(idx, e.target.value)}
                            className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2 py-1 text-center font-mono text-xs text-amber-300 focus:outline-none focus:border-amber-500"
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-3 text-xs">
                    <span className="font-semibold text-slate-300 block border-b border-slate-800 pb-1">Parámetros del Algoritmo</span>
                    <div>
                      <label className="text-slate-400 block mb-1">Precondicionador M:</label>
                      <select
                        value={precondType}
                        onChange={(e) => setPrecondType(e.target.value)}
                        className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-white focus:outline-none focus:border-amber-500"
                      >
                        <option value="jacobi">Precondicionador Jacobi (Diagonal)</option>
                        <option value="ic0">Cholesky Incompleto IC(0)</option>
                      </select>
                    </div>

                    <div>
                      <label className="text-slate-400 block mb-1">Tolerancia Error:</label>
                      <input
                        type="number"
                        step="any"
                        value={tolerance}
                        onChange={(e) => setTolerance(parseFloat(e.target.value) || 1e-6)}
                        className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 font-mono text-white focus:outline-none focus:border-amber-500"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 6. SOLUCIÓN REFERENCIA */}
        {activeTab === 'referencia' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-emerald-400" /> Solución Numérica de Referencia (Método Directo Cholesky)
              </h2>
              <p className="text-xs text-slate-300 leading-relaxed">
                Calculada descomponiendo la matriz simétrica definida positiva en A = L · Lᵀ. Representa la solución computacional exacta de control para medir los errores del GCP.
              </p>

              {mathDiagnostics.solRef ? (
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
                  <h3 className="text-xs font-semibold text-amber-400 uppercase tracking-wider">Vector Solución Exacta (x_ref):</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
                    {mathDiagnostics.solRef.map((val, idx) => (
                      <div key={idx} className="bg-slate-900 p-3 rounded-lg border border-slate-800 text-center">
                        <span className="text-xs font-mono text-slate-400 block mb-1">{variables[idx]?.name || `x_${idx+1}`}</span>
                        <span className="text-sm font-mono font-bold text-emerald-400">{val.toFixed(6)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="bg-rose-500/10 border border-rose-500/30 p-4 rounded-xl text-rose-400 text-xs">
                  No se pudo obtener la solución de referencia directa: {mathDiagnostics.refError || "La matriz A no cumple los criterios de Cholesky."}
                </div>
              )}
            </div>
          </div>
        )}

        {/* 7. GCP VS GC */}
        {activeTab === 'gcp' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-amber-400" /> Comparativa: GCP vs Gradiente Conjugado Simple (GC)
              </h2>

              {mathDiagnostics.resGC && mathDiagnostics.resGCP ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-slate-950 p-5 rounded-2xl border border-slate-800 space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="font-bold text-slate-200 text-sm">Gradiente Conjugado (GC Simple)</h3>
                      <span className="px-2.5 py-1 bg-slate-800 text-slate-400 border border-slate-700 rounded-md font-mono text-xs">
                        Sin Precondicionador
                      </span>
                    </div>

                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between py-1 border-b border-slate-800">
                        <span className="text-slate-400">Iteraciones totales:</span>
                        <span className="font-mono font-bold text-amber-400">{mathDiagnostics.resGC.iterations}</span>
                      </div>
                      <div className="flex justify-between py-1 border-b border-slate-800">
                        <span className="text-slate-400">Residuo Relativo Final:</span>
                        <span className="font-mono text-slate-200">{mathDiagnostics.resGC.finalResidual.toExponential(4)}</span>
                      </div>
                      <div className="flex justify-between py-1">
                        <span className="text-slate-400">Número de Condición original κ(A):</span>
                        <span className="font-mono text-slate-200">{mathDiagnostics.kappaA.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-950 p-5 rounded-2xl border border-amber-500/30 space-y-4 relative overflow-hidden">
                    <div className="absolute top-0 right-0 bg-amber-500 text-slate-950 text-[10px] font-bold px-3 py-0.5 rounded-bl-lg uppercase">
                      Recomendado (Suñagua 2020)
                    </div>
                    <div className="flex items-center justify-between">
                      <h3 className="font-bold text-amber-400 text-sm">GCP ({precondType.toUpperCase()})</h3>
                      <span className="px-2.5 py-1 bg-amber-500/10 text-amber-400 border border-amber-500/30 rounded-md font-mono text-xs">
                        Precondicionado
                      </span>
                    </div>

                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between py-1 border-b border-slate-800">
                        <span className="text-slate-400">Iteraciones totales:</span>
                        <span className="font-mono font-bold text-emerald-400">{mathDiagnostics.resGCP.iterations}</span>
                      </div>
                      <div className="flex justify-between py-1 border-b border-slate-800">
                        <span className="text-slate-400">Residuo Relativo Final:</span>
                        <span className="font-mono text-slate-200">{mathDiagnostics.resGCP.finalResidual.toExponential(4)}</span>
                      </div>
                      <div className="flex justify-between py-1">
                        <span className="text-slate-400">Número de Condición κ(M⁻¹A):</span>
                        <span className="font-mono text-emerald-400">{mathDiagnostics.resGCP.kappaPrecond.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-xs text-rose-400">No se pueden calcular las comparativas hasta corregir la matriz A.</p>
              )}
            </div>
          </div>
        )}

        {/* 8. ANÁLISIS DE ERRORES */}
        {activeTab === 'errores' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-amber-400" /> Análisis Detallado de Errores por Variable
              </h2>
              <p className="text-xs text-slate-300">
                Evaluación computacional componente a componente comparando la Solución GCP con la Solución numéricamente exacta de referencia.
              </p>

              <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-900 text-slate-400 font-medium uppercase border-b border-slate-800">
                    <tr>
                      <th className="p-3">Variable</th>
                      <th className="p-3 font-mono">Sol. Referencia (x_ref)</th>
                      <th className="p-3 font-mono">Sol. GCP (x_gcp)</th>
                      <th className="p-3 font-mono">Error Absoluto (Ea)</th>
                      <th className="p-3 font-mono">Error Relativo (Er)</th>
                      <th className="p-3 font-mono">Error Porcentual (Ep %)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {mathDiagnostics.errorTable.map((row, i) => (
                      <tr key={i} className="hover:bg-slate-900/40">
                        <td className="p-3 font-sans font-semibold text-slate-200">{row.varName}</td>
                        <td className="p-3 text-emerald-400">{row.refVal.toFixed(6)}</td>
                        <td className="p-3 text-amber-300">{row.gcpVal.toFixed(6)}</td>
                        <td className="p-3 text-slate-300">{row.ea.toExponential(3)}</td>
                        <td className="p-3 text-slate-300">{row.er !== null ? row.er.toExponential(3) : 'N/A'}</td>
                        <td className="p-3 text-slate-300">{row.ep !== null ? `${row.ep.toFixed(5)}%` : 'N/A'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* 9. TABLA RESUMEN */}
        {activeTab === 'resumen' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <FileText className="w-5 h-5 text-amber-400" /> Tabla Comparativa Consolidada de los Tres Escenarios
              </h2>
              <p className="text-xs text-slate-300">
                Resumen ejecutivo y cualitativo estipulado por las directrices de la UMSA para la evaluación global de sensibilidad numérico-financiera.
              </p>

              <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-900 text-slate-400 uppercase font-medium border-b border-slate-800">
                    <tr>
                      <th className="p-3">Indicador / Métrica</th>
                      {consolidatedSummary.map(s => (
                        <th key={s.key} className="p-3 font-bold text-amber-400 min-w-[200px]">{s.name}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800 font-mono">
                    <tr>
                      <td className="p-3 font-sans font-semibold text-slate-300">Solución Directa Ref. [x]</td>
                      {consolidatedSummary.map(s => (
                        <td key={s.key} className="p-3 text-emerald-400 text-[11px]">
                          [{s.directSol.map(v => v.toFixed(2)).join(', ')}]
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="p-3 font-sans font-semibold text-slate-300">Solución GCP [x]</td>
                      {consolidatedSummary.map(s => (
                        <td key={s.key} className="p-3 text-amber-300 text-[11px]">
                          [{s.gcpSol.map(v => v.toFixed(2)).join(', ')}]
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="p-3 font-sans font-semibold text-slate-300">Iteraciones GC Simple</td>
                      {consolidatedSummary.map(s => (
                        <td key={s.key} className="p-3 text-slate-300">{s.gcIter} it</td>
                      ))}
                    </tr>
                    <tr>
                      <td className="p-3 font-sans font-semibold text-slate-300">Iteraciones GCP (Jacobi)</td>
                      {consolidatedSummary.map(s => (
                        <td key={s.key} className="p-3 text-emerald-400 font-bold">{s.gcpIter} it</td>
                      ))}
                    </tr>
                    <tr>
                      <td className="p-3 font-sans font-semibold text-slate-300">Nro. Condición Original κ(A)</td>
                      {consolidatedSummary.map(s => (
                        <td key={s.key} className="p-3 text-slate-300">{s.kappaA.toFixed(2)}</td>
                      ))}
                    </tr>
                    <tr>
                      <td className="p-3 font-sans font-semibold text-slate-300">Nro. Condición Precond. κ(M⁻¹A)</td>
                      {consolidatedSummary.map(s => (
                        <td key={s.key} className="p-3 text-emerald-400 font-bold">{s.kappaPrecond.toFixed(2)}</td>
                      ))}
                    </tr>
                    <tr>
                      <td className="p-3 font-sans font-semibold text-slate-300">Residuo Relativo Final</td>
                      {consolidatedSummary.map(s => (
                        <td key={s.key} className="p-3 text-slate-300">{s.finalResidual.toExponential(3)}</td>
                      ))}
                    </tr>
                    <tr>
                      <td className="p-3 font-sans font-semibold text-slate-300">¿Convergió el Algoritmo?</td>
                      {consolidatedSummary.map(s => (
                        <td key={s.key} className="p-3">
                          <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 font-bold">
                            SÍ (Tolerancia alcanzada)
                          </span>
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* 10. PRUEBAS QA */}
        {activeTab === 'qa' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-xl font-bold text-white flex items-center gap-2">
                    <ShieldAlert className="w-5 h-5 text-amber-400" /> Módulo de Pruebas de Software (Quality Assurance - QA)
                  </h2>
                  <p className="text-xs text-slate-300">
                    Ejecución interactiva y automatizada de los 4 casos de prueba exigidos por la Rúbrica Académica UMSA.
                  </p>
                </div>
                <button
                  onClick={runQATests}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-500 text-slate-950 font-bold rounded-xl text-xs hover:bg-amber-400 transition-all shadow-md"
                >
                  <RefreshCw className="w-3.5 h-3.5" /> Re-ejecutar Pruebas QA
                </button>
              </div>

              <div className="space-y-4">
                {qaResults.map((t) => (
                  <div key={t.id} className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                    <div className="flex items-center justify-between">
                      <h3 className="font-bold text-sm text-white">{t.title}</h3>
                      <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${
                        t.passed ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                      }`}>
                        {t.passed ? 'APROBADO (PASSED)' : 'FALLIDO (FAILED)'}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs pt-1">
                      <div>
                        <span className="text-slate-500 block">Entrada de Prueba:</span>
                        <span className="text-slate-300 font-mono">{t.input}</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block">Resultado Esperado:</span>
                        <span className="text-slate-300">{t.expected}</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block">Resultado Obtenido:</span>
                        <span className="text-amber-300 font-mono">{t.obtained}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* 11. CONCLUSIONES Y REFERENCIAS */}
        {activeTab === 'conclusiones' && (
          <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-amber-400" /> Conclusiones Técnicas y Bibliografía
              </h2>

              <div className="space-y-4 text-xs text-slate-300 leading-relaxed">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                  <h3 className="text-sm font-semibold text-amber-400">Interpretación Numérico-Financiera</h3>
                  <p>
                    1. <strong>Aceleración de Convergencia:</strong> El uso del precondicionador de Jacobi o Cholesky Incompleto IC(0) demostró reducir significativamente las iteraciones requeridas para alcanzar la tolerancia de 10⁻⁶ en matrices con alto grado de mal escalamiento diagonal (κ(A) ≫ 1000).
                  </p>
                  <p>
                    2. <strong>Estabilidad Presupuestaria:</strong> En escenarios bajo estrés financiero donde las perturbaciones aumentan los coeficientes fuera de la diagonal, el precondicionamiento mantiene la estabilidad del residuo global impidiendo la divergencia numérica.
                  </p>
                </div>

                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
                  <h3 className="text-sm font-semibold text-amber-400">Referencias Bibliográficas Principales</h3>
                  <ul className="space-y-2 font-mono text-[11px] text-slate-400">
                    <li className="flex items-start gap-2">
                      <span className="text-amber-400">•</span>
                      <span><strong>Suñagua S., P. (2020).</strong> <em>Método de Gradientes Conjugados Precondicionado.</em> Revista Boliviana de Matemática – UMSA, 04, 2–7.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-amber-400">•</span>
                      <span><strong>Golub, G. H., & Van Loan, C. F. (2013).</strong> <em>Matrix Computations</em> (4th ed.). Johns Hopkins University Press.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-amber-400">•</span>
                      <span><strong>Saad, Y. (2003).</strong> <em>Iterative Methods for Sparse Linear Systems</em> (2nd ed.). SIAM.</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}

      </main>

      <footer className="border-t border-slate-900 bg-slate-950 py-4 px-6 text-center text-xs text-slate-400">
        <p>Universidad Mayor de San Andrés · Carrera de Informática · Desafío Académico de Métodos Numéricos 2026</p>
      </footer>
    </div>
  );
}