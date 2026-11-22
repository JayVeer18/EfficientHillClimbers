package neo.landscape.theory.apps.pseudoboolean.px;

import neo.landscape.theory.apps.pseudoboolean.PBSolution;
import neo.landscape.theory.apps.pseudoboolean.problems.MultiObjectiveLandscape;
import neo.landscape.theory.apps.pseudoboolean.util.NonDominatedSet;

import java.util.Arrays;
import java.util.List;

/**
 * <h1>Transition Q Partition Crossover (Transition-QPX / MOPX)</h1>
 *
 * <p>
 * Implements an exact, zero-crossing critical weight transition sweep for Multi-Objective
 * Partition Crossover across an arbitrary number of objectives <i>M &ge; 2</i>.
 * Integrates directly with {@link neo.landscape.theory.apps.pseudoboolean.util.NonDominatedSet}
 * to archive supported Pareto-optimal offspring discovered along simplex rays.
 * </p>
 *
 * <h2>1. Theoretical Foundations &amp; Research Context</h2>
 * <p>
 * In Partition Crossover (PX), crossing two parents <i>P<sub>blue</sub></i> and <i>P<sub>red</sub></i>
 * decomposes into <i>q</i> independent connected components of differing variables on the reduced graph.
 * This defines a recombination hypercube of <i>2<sup>q</sup></i> candidate offspring.
 * </p>
 * <ul>
 *   <li><b>Evostar 2025 / GECCO 2027 Context:</b> Developed based on the collaboration between Darrell Whitley,
 *       Francisco Chicano, and Surendra Kurivella exploring whether critical weight zero-crossings can efficiently
 *       harvest the supported upper convex hull of PX offspring without exponential enumeration.</li>
 *   <li><b>Standard Multi-Objective PX (QPX):</b> Evaluates the single best child under active scalarization
 *       weights <i>w</i> plus its <i>q</i> Hamming-1 neighbors. While linear in cost <i>O(q)</i>, empirical evidence
 *       shows that Hamming-1 neighbors cluster tightly around the scalarized best child rather than spreading
 *       broadly across the Pareto front.</li>
 *   <li><b>Transition-QPX (MOPX):</b> Rather than sampling arbitrary weight steps or clustered Hamming-1 neighbors,
 *       this operator identifies the <b>exact zero-crossing thresholds</b> where the optimal parent choice
 *       for each component flips. Each transition point in <i>(0, 1)</i> produces a <b>distinct supported
 *       non-dominated offspring</b> on the crossover lattice in <i>O(M &middot; q log q)</i> total time.</li>
 * </ul>
 *
 * <h2>2. Mathematical Formulation &amp; Zero-Crossing Derivation</h2>
 *
 * <h3>2.1 Component Differentials</h3>
 * <p>
 * For each independent component <i>c &isin; {0, &hellip;, q-1}</i>, the raw objective differential vector is:
 * </p>
 * <pre>
 *   &Delta;f[c] = f_blue[c] - f_red[c] = (&Delta;f_1[c], &Delta;f_2[c], ..., &Delta;f_M[c])
 * </pre>
 *
 * <h3>2.2 Simplex Directional Rays</h3>
 * <p>
 * On the unit weight simplex (&sum; w_m = 1, w_m &ge; 0),
 * exploring from the active search vector <i>w</i> towards any basis vertex <i>e_m</i>
 * follows the parameterized ray:
 * </p>
 * <pre>
 *   w(t) = (1 - t) w + t e_m,  where t &isin; [0, 1]
 * </pre>
 * <p>
 * Under parameter <i>t</i>, the scalarized contribution of component <i>c</i> is strictly linear:
 * </p>
 * <pre>
 *   &Delta;(t; c) = (1 - t) D_0[c] + t D_m[c]
 * </pre>
 * <p>
 * where:
 * </p>
 * <pre>
 *   D_0[c] = w &middot; &Delta;f[c] = &sum; w_j &Delta;f_j[c]
 *   D_m[c] = e_m &middot; &Delta;f[c] = &Delta;f_m[c]
 * </pre>
 *
 * <h3>2.3 Exact Zero-Crossing Thresholds</h3>
 * <p>
 * Component <i>c</i> transitions between parent procedences when its marginal scalar contribution passes through zero:
 * </p>
 * <pre>
 *   &Delta;(t_c*; c) = 0 &rArr; (1 - t_c*) D_0[c] + t_c* D_m[c] = 0
 *   &rArr; t_c* = D_0[c] / (D_0[c] - D_m[c])
 * </pre>
 *
 * <h3>2.4 Guaranteed Boundedness in (0, 1) &amp; Resolution of Darrell's Spreadsheet Dilemma</h3>
 * <p>
 * In Darrell Whitley's spreadsheet (<i>EQUALITIES of weight and alpha copy.xlsx</i>, row 37: <i>WEIGHTS without A0, a0</i>),
 * the calculation for Component 3 (with <i>A<sub>3</sub> = 4, a<sub>3</sub> = -6</i>) was written as:
 * </p>
 * <pre>
 *   w = a_3 / (a_3 + A_3) = -6 / (-6 + 4) = 3  (annotated as "NOT FRACTIONAL")
 * </pre>
 * <p>
 * The unexpected value <i>w = 3</i> occurred due to using addition in the denominator instead of subtraction from the zero-crossing equality:
 * </p>
 * <pre>
 *   w* A_3 + (1 - w*) a_3 = 0
 *   &rArr; w* (A_3 - a_3) = -a_3
 *   &rArr; w* = a_3 / (a_3 - A_3) = -6 / (-6 - 4) = 0.60 &isin; (0, 1)
 * </pre>
 * <p>
 * Similarly for Component 2 (<i>A<sub>2</sub> = -12, a<sub>2</sub> = 8</i>):
 * </p>
 * <pre>
 *   w* = a_2 / (a_2 - A_2) = 8 / (8 - (-12)) = 8 / 20 = 0.40 &isin; (0, 1)
 * </pre>
 * <p>
 * <b>General Boundedness Proof:</b> For any trade-off component, <i>D<sub>0</sub>[c]</i> and <i>D<sub>m</sub>[c]</i> have opposite signs.
 * Therefore, <i>D<sub>0</sub>[c] - D<sub>m</sub>[c] = |D<sub>0</sub>[c]| + |D<sub>m</sub>[c]| &gt; |D<sub>0</sub>[c]|</i>.
 * It follows that:
 * </p>
 * <pre>
 *   0 &lt; t_c* = |D_0[c]| / (|D_0[c]| + |D_m[c]|) &lt; 1
 * </pre>
 * <p>
 * Thus, all trade-off transitions are <b>strictly bounded in (0, 1)</b> without requiring any reference solution <i>P<sub>0</sub></i>.
 * Components where <i>D<sub>0</sub>[c]</i> and <i>D<sub>m</sub>[c]</i> share the same sign are unanimous (strictly dominated)
 * and have no zero-crossing in (0, 1).
 * </p>
 *
 * <h3>2.5 Dimensional Universality (M = 2 vs. M &ge; 3)</h3>
 * <ul>
 *   <li><b>Bi-objective (M = 2):</b> The two basis rays towards <i>e<sub>1</sub> = (1, 0)</i> and <i>e<sub>2</sub> = (0, 1)</i>
 *       cover the exact 1D line segment [0, 1]. Sorting these transitions in <i>O(q log q)</i> yields the complete
 *       supported upper convex hull of the <i>2<sup>q</sup></i> offspring space.</li>
 *   <li><b>Many-objective (M &ge; 3):</b> The <i>M</i> basis rays discover trade-off transitions along all simplex
 *       axes in <i>O(M &middot; q log q)</i> time, capturing non-clustered, supported offspring across the Pareto surface
 *       without requiring exponential enumeration of higher-dimensional simplex facets.</li>
 * </ul>
 *
 * <h2>3. Darrell's Micro-Sweep Windowing (&delta;)</h2>
 * <p>
 * Supports localized search around the active search trajectory <i>w</i>:
 * </p>
 * <ul>
 *   <li>{@code delta <= 0.0}: Performs a full ray sweep over <i>t &isin; [0, 1]</i>.</li>
 *   <li>{@code delta > 0.0}: Restricts evaluations to <i>t &le; delta</i>, focusing offspring sampling
 *       within the immediate neighborhood of <i>w</i> to preserve search locality.</li>
 * </ul>
 *
 * <h2>4. Zero-Allocation Architecture &amp; Performance</h2>
 * <p>
 * All primitive working arrays ({@code deltaRawObjsFlat}, {@code compD0}, {@code candidateBuffer},
 * {@code currentWeightsBuffer}, {@code tradeOffCompIndex}, {@code transitionWeights},
 * {@code blueValBuffer}, {@code redValBuffer}) are pre-allocated during construction.
 * Sorting is performed via an in-place primitive quicksort.
 * This guarantees zero garbage collection overhead in the high-frequency evaluation loops of DRILS.
 * </p>
 *
 * <h2>5. CLI Integration &amp; Execution</h2>
 * <pre>
 *   java -jar McDRILSMO.jar mcdrilsmo -crossover mopx -delta 0.05 -r 1 -mf 0.2 -P objs=2 ...
 * </pre>
 *
 * @author Darrell Whitley, Francisco Chicano, Surendra Kurivella
 * @version 1.0
 */
public class TransitionQPartitionCrossover extends PartitionCrossover {

    private final MultiObjectiveLandscape moLandscape;
    private NonDominatedSet archive;

    private final int numObjectives;
    private final int[] objectiveMap;

    // Zero-allocation evaluation buffers
    private final double[] blueValBuffer;
    private final double[] redValBuffer;
    private final boolean[] affectedSubfnsBuffer;
    private final int[] touchedSubfnsBuffer;

    // Pre-allocated component differentials (flattened: [maxN * numObjectives])
    private final PartitionComponent[] compArray;
    private final double[] deltaRawObjsFlat;
    private final double[] compD0;
    private final double[] bestRawObjs;
    private final double[] candidateBuffer;
    private final double[] currentWeightsBuffer;

    // Ray transition sorting and tracking buffers
    private final int[] tradeOffCompIndex;
    private final double[] transitionWeights;

    // Micro-sweep window delta (0.0: full ray sweep [0, 1]; > 0.0: restricted to [0, delta])
    private double delta = 0.0;

    /**
     * Constructs a TransitionQPartitionCrossover operator for the given landscape.
     *
     * @param el the multi-objective problem landscape (M &ge; 2)
     */
    public TransitionQPartitionCrossover(MultiObjectiveLandscape el) {
        super(el);
        this.moLandscape = el;
        this.numObjectives = el.getNumObjectives();

        if (numObjectives < 2) {
            throw new IllegalArgumentException("TransitionQPartitionCrossover requires at least 2 objectives.");
        }

        int maxN = el.getN();
        this.compArray = new PartitionComponent[maxN];
        this.deltaRawObjsFlat = new double[maxN * numObjectives];
        this.compD0 = new double[maxN];
        this.bestRawObjs = new double[numObjectives];
        this.candidateBuffer = new double[numObjectives];
        this.currentWeightsBuffer = new double[numObjectives];

        this.tradeOffCompIndex = new int[maxN];
        this.transitionWeights = new double[maxN];

        this.blueValBuffer = new double[numObjectives];
        this.redValBuffer = new double[numObjectives];
        this.affectedSubfnsBuffer = new boolean[el.getM()];
        this.touchedSubfnsBuffer = new int[el.getM()];

        this.objectiveMap = new int[el.getM()];
        List<Integer> mValues = el.getMValues();
        int currentObj = 0;
        int nextBoundary = mValues.get(0);

        for (int i = 0; i < el.getM(); i++) {
            if (i >= nextBoundary && currentObj < numObjectives - 1) {
                currentObj++;
                nextBoundary += mValues.get(currentObj);
            }
            objectiveMap[i] = currentObj;
        }
    }

    /**
     * Sets the external non-dominated archive for collecting supported Pareto-optimal solutions.
     *
     * @param archive the Pareto archive (e.g., NonDominatedSet)
     */
    public void setArchive(NonDominatedSet archive) {
        this.archive = archive;
    }

    /**
     * Returns the active micro-sweep delta window size.
     *
     * @return delta window size (0.0 means full sweep)
     */
    public double getDelta() {
        return delta;
    }

    /**
     * Configures the micro-sweep delta window size.
     *
     * @param delta half-width of the window; if &le; 0.0, a full ray sweep [0, 1] is performed
     */
    public void setDelta(double delta) {
        this.delta = delta;
    }

    /**
     * Recombines two parent solutions, evaluating critical transitions, archiving supported offspring,
     * and returning the scalarized optimal child under the active weights.
     *
     * @param blue the blue parent
     * @param red  the red parent
     * @return the recombined child solution
     */
    @Override
    public PBSolution recombine(PBSolution blue, PBSolution red) {
        return recombineInternal(blue, red);
    }

    /**
     * Internal recombination logic decomposing the reduced graph into independent partition components,
     * executing ray transition sweeps to archive non-dominated solutions, and assembling the child.
     *
     * @param blue the blue parent
     * @param red  the red parent
     * @return the recombined child solution
     */
    @Override
    public PBSolution recombineInternal(PBSolution blue, PBSolution red) {
        long initTime = System.nanoTime();
        bfsSet.reset();

        PBSolution child = new PBSolution(red);
        numberOfComponents = 0;

        // Cache active weights without array allocation
        for (int m = 0; m < numObjectives; m++) {
            currentWeightsBuffer[m] = moLandscape.getWeight(m);
        }

        // 1. Decompose into independent partition components and compute differentials
        for (Integer node = nextNodeInReducedGraph(blue, red); node != null; node = nextNodeInReducedGraph(blue, red)) {
            PartitionComponent component = bfs(node, blue, red);
            compArray[numberOfComponents] = component;

            evaluateComponentWeightedObjs(component, blue, blueValBuffer);
            evaluateComponentWeightedObjs(component, red, redValBuffer);

            int flatOffset = numberOfComponents * numObjectives;
            double d0 = 0.0;
            for (int m = 0; m < numObjectives; m++) {
                double wDiff = blueValBuffer[m] - redValBuffer[m];
                d0 += wDiff;
                double w = currentWeightsBuffer[m];
                deltaRawObjsFlat[flatOffset + m] = (w > 1e-9) ? wDiff / w : wDiff;
            }
            compD0[numberOfComponents] = d0;

            numberOfComponents++;
        }

        // 2. Perform transition sweeps and archive supported non-dominated offspring
        if (archive != null) {
            sweepTransitions(red);
        }

        // 3. Assemble and return the scalarized optimal child for the active search weights
        for (int c = 0; c < numberOfComponents; c++) {
            double d0 = compD0[c];
            boolean pickBlue = (d0 > 0) || ((d0 == 0) && rnd.nextDouble() < 0.5);
            if (pickBlue) {
                PartitionComponent comp = compArray[c];
                for (int var : comp) {
                    child.setBit(var, blue.getBit(var));
                    varProcedence.markAsBlue(var);
                }
            }
        }

        lastRuntime = System.nanoTime() - initTime;
        return child;
    }

    /**
     * Sweeps exact critical transition points along directional rays on the simplex.
     *
     * @param red the baseline parent solution
     */
    private void sweepTransitions(PBSolution red) {
        evaluateFullRawObjs(red, bestRawObjs);

        // A. Baseline: compute and archive the optimal solution under currentWeights
        for (int c = 0; c < numberOfComponents; c++) {
            if (compD0[c] > 0) {
                int flatOffset = c * numObjectives;
                for (int m = 0; m < numObjectives; m++) {
                    bestRawObjs[m] += deltaRawObjsFlat[flatOffset + m];
                }
            }
        }
        archive.reportSolutionToArchive(bestRawObjs);

        // B. QPX Baseline: archive Hamming-1 neighbor deviations using zero-allocation buffer
        for (int c = 0; c < numberOfComponents; c++) {
            int flatOffset = c * numObjectives;
            double sign = (compD0[c] > 0) ? -1.0 : 1.0;
            for (int m = 0; m < numObjectives; m++) {
                candidateBuffer[m] = bestRawObjs[m] + sign * deltaRawObjsFlat[flatOffset + m];
            }
            archive.reportSolutionToArchive(candidateBuffer);
        }

        // C. Directional Transition Sweeps towards each simplex basis vertex e_target
        double maxT = (delta > 0.0) ? Math.min(1.0, delta) : 1.0;

        for (int targetM = 0; targetM < numObjectives; targetM++) {
            int rayTradeOffCount = 0;

            for (int c = 0; c < numberOfComponents; c++) {
                double d0 = compD0[c];
                int flatOffset = c * numObjectives;
                double dm = deltaRawObjsFlat[flatOffset + targetM];

                // If signs differ, there is an exact zero-crossing point in (0, 1)
                if ((d0 > 0 && dm < 0) || (d0 < 0 && dm > 0)) {
                    double tStar = d0 / (d0 - dm);
                    if (tStar > 0.0 && tStar <= maxT) {
                        tradeOffCompIndex[rayTradeOffCount] = c;
                        transitionWeights[rayTradeOffCount] = tStar;
                        rayTradeOffCount++;
                    }
                }
            }

            if (rayTradeOffCount > 0) {
                quickSort(tradeOffCompIndex, transitionWeights, 0, rayTradeOffCount - 1);

                System.arraycopy(bestRawObjs, 0, candidateBuffer, 0, numObjectives);
                for (int i = 0; i < rayTradeOffCount; i++) {
                    int c = tradeOffCompIndex[i];
                    int flatOffset = c * numObjectives;
                    double flipSign = (compD0[c] > 0) ? -1.0 : 1.0;
                    for (int m = 0; m < numObjectives; m++) {
                        candidateBuffer[m] += flipSign * deltaRawObjsFlat[flatOffset + m];
                    }
                    archive.reportSolutionToArchive(candidateBuffer);
                }
            }
        }
    }

    /**
     * In-place primitive quicksort to sort component indices in ascending order of transition weights.
     * Operates directly on pre-allocated primitive arrays with zero memory allocation.
     *
     * @param indices component indices to sort
     * @param weights transition weights used as sorting keys
     * @param low     starting index
     * @param high    ending index
     */
    private void quickSort(int[] indices, double[] weights, int low, int high) {
        if (low < high) {
            int pi = partition(indices, weights, low, high);
            quickSort(indices, weights, low, pi - 1);
            quickSort(indices, weights, pi + 1, high);
        }
    }

    /**
     * Lomuto partition scheme for in-place quicksort.
     */
    private int partition(int[] indices, double[] weights, int low, int high) {
        double pivot = weights[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (weights[j] <= pivot) {
                i++;
                swap(indices, weights, i, j);
            }
        }
        swap(indices, weights, i + 1, high);
        return i + 1;
    }

    /**
     * Swaps elements at positions i and j in both indices and weights arrays.
     */
    private void swap(int[] indices, double[] weights, int i, int j) {
        int tempIdx = indices[i];
        indices[i] = indices[j];
        indices[j] = tempIdx;

        double tempW = weights[i];
        weights[i] = weights[j];
        weights[j] = tempW;
    }

    /**
     * Evaluates component sub-functions directly into a reusable primitive buffer.
     * Uses primitive boolean array deduplication to prevent boxed object allocation.
     *
     * @param vars      variables belonging to the component
     * @param sol       solution to evaluate
     * @param outBuffer reusable buffer where component objective contributions are accumulated
     */
    private void evaluateComponentWeightedObjs(Iterable<Integer> vars, PBSolution sol, double[] outBuffer) {
        Arrays.fill(outBuffer, 0.0);
        int touchedCount = 0;

        for (int variable : vars) {
            for (int fn : el.getAppearsIn()[variable]) {
                if (!affectedSubfnsBuffer[fn]) {
                    affectedSubfnsBuffer[fn] = true;
                    touchedSubfnsBuffer[touchedCount++] = fn;
                }
            }
        }

        for (int i = 0; i < touchedCount; i++) {
            int fn = touchedSubfnsBuffer[i];
            double val = el.evaluateSubFunctionFromCompleteSolution(fn, sol);
            int objIndex = objectiveMap[fn];
            outBuffer[objIndex] += val;
            affectedSubfnsBuffer[fn] = false;
        }
    }

    /**
     * Evaluates unweighted raw objective values of a complete solution.
     * Computes the unweighted vector f(sol) = (f_1(sol), ..., f_M(sol)) into the reusable buffer.
     *
     * @param sol       complete candidate solution
     * @param outBuffer reusable buffer where raw objective values are stored
     */
    private void evaluateFullRawObjs(PBSolution sol, double[] outBuffer) {
        Arrays.fill(outBuffer, 0.0);

        for (int i = 0; i < el.getM(); i++) {
            double val = el.evaluateSubFunctionFromCompleteSolution(i, sol);
            int objIndex = objectiveMap[i];
            outBuffer[objIndex] += val;
        }

        for (int o = 0; o < numObjectives; o++) {
            double w = currentWeightsBuffer[o];
            if (w > 1e-9) {
                outBuffer[o] /= w;
            }
        }
    }
}
