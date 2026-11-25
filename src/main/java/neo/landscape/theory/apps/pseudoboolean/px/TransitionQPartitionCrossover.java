package neo.landscape.theory.apps.pseudoboolean.px;

import neo.landscape.theory.apps.pseudoboolean.PBSolution;
import neo.landscape.theory.apps.pseudoboolean.problems.MultiObjectiveLandscape;
import neo.landscape.theory.apps.pseudoboolean.util.NonDominatedSet;

import java.util.Arrays;
import java.util.List;

/**
 * Multi-Objective Partition Crossover with Critical Weight Transitions (MOPX).
 *
 * <p>
 * Given two parent solutions (blue and red), Partition Crossover decomposes
 * the recombination space into q independent connected components on the
 * reduced variable interaction graph, inducing a space of 2^q potential offspring.
 * </p>
 *
 * <p>
 * Standard QPX evaluates the scalarized best child under the current weight
 * vector w and samples its q 1-Hamming distance neighbors.
 * MOPX instead computes the exact critical transition weights where component
 * choices flip along directional rays on the weight simplex.
 * </p>
 *
 * <h3>Critical Weight Transitions</h3>
 * <p>
 * For each component c, let &Delta;f[c] = f_blue[c] - f_red[c] be the raw objective
 * differential vector. Under a parameterized ray w(t) = (1 - t) w + t e_m connecting
 * the active search direction w to basis vertex e_m:
 * </p>
 * <pre>
 *   &Delta;Fitness(t; c) = (1 - t) D_0[c] + t D_m[c]
 * </pre>
 * <p>
 * where D_0[c] = &sum; w_j &Delta;f_j[c] is the current scalar contribution,
 * and D_m[c] = &Delta;f_m[c] is the contribution towards objective m.
 * Setting &Delta;Fitness(t_c*; c) = 0 gives the critical transition parameter:
 * </p>
 * <pre>
 *   t_c* = |D_0[c]| / (|D_0[c]| + |D_m[c]|)
 * </pre>
 * <p>
 * For components representing a trade-off (D_0 and D_m have opposite signs),
 * t_c* is strictly in (0, 1). Sorting these transitions in O(q log q) time
 * traces the upper convex hull of supported offspring along each ray.
 * </p>
 *
 * <h3>Micro-sweep Window (delta)</h3>
 * <p>
 * The parameter delta restricts the transition sweep to t &le; delta (0.0 implies
 * a full ray sweep over [0, 1]).
 * </p>
 */
public class TransitionQPartitionCrossover extends PartitionCrossover {

    private final MultiObjectiveLandscape moLandscape;
    private NonDominatedSet archive;

    private final int numObjectives;
    private final int[] objectiveMap;

    // Working buffers for component evaluations
    private final double[] blueValBuffer;
    private final double[] redValBuffer;
    private final boolean[] affectedSubfnsBuffer;
    private final int[] touchedSubfnsBuffer;

    // Component differentials and scalar contributions
    private final boolean[] pickedBlue;
    private final double[] deltaRawObjsFlat;
    private final double[] compD0;
    private final double[] bestRawObjs;
    private final double[] candidateBuffer;
    private final double[] currentWeightsBuffer;

    // Ray transition sorting and tracking buffers
    private final int[] tradeOffCompIndex;
    private final double[] transitionWeights;

    // Window limit for transition sweep (0.0: full sweep [0, 1])
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
        this.pickedBlue = new boolean[maxN];
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

            // Assemble optimal scalarized child bits and procedence directly while component is active
            boolean pickBlue = (d0 > 0) || ((d0 == 0) && rnd.nextDouble() < 0.5);
            pickedBlue[numberOfComponents] = pickBlue;
            if (pickBlue) {
                for (int var : component) {
                    child.setBit(var, blue.getBit(var));
                    varProcedence.markAsBlue(var);
                }
            }

            numberOfComponents++;
        }

        // 2. Perform transition sweeps and archive supported non-dominated offspring
        if (archive != null) {
            sweepTransitions(red);
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

        // A. Baseline: compute and archive the exact optimal solution for active weights
        for (int c = 0; c < numberOfComponents; c++) {
            if (pickedBlue[c]) {
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
            double sign = pickedBlue[c] ? -1.0 : 1.0;
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

                // If flipping component improves targetM:
                // If Blue is currently picked, flipping to Red improves targetM iff dm < 0
                // If Red is currently picked, flipping to Blue improves targetM iff dm > 0
                boolean canImproveTarget = (pickedBlue[c] && dm < 0) || (!pickedBlue[c] && dm > 0);
                if (canImproveTarget && Math.abs(d0) > 1e-9) {
                    double absD0 = Math.abs(d0);
                    double absDm = Math.abs(dm);
                    double tStar = absD0 / (absD0 + absDm);
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
                    double flipSign = pickedBlue[c] ? -1.0 : 1.0;
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
