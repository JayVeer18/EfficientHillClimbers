package neo.landscape.theory.apps.pseudoboolean.px;

import neo.landscape.theory.apps.pseudoboolean.PBSolution;
import neo.landscape.theory.apps.pseudoboolean.problems.EmbeddedLandscape;
import neo.landscape.theory.apps.pseudoboolean.problems.MultiObjectiveLandscape;
import neo.landscape.theory.apps.pseudoboolean.problems.NKLandscapes;
import neo.landscape.theory.apps.pseudoboolean.util.NonDominatedSet;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Properties;

/**
 * Empirical Pareto Front Coverage, Spread, and Solution Quality Benchmark: QPX vs. MOPX.
 * Evaluates:
 * 1. Hypervolume (HV) relative to reference point.
 * 2. Objective ranges (Spread in f1 and f2).
 * 3. Mutual dominance (Number of solutions dominated by the opposing operator).
 * 4. Front diversity and Euclidean span from the scalarized center.
 */
public class ParetoQualityBenchmark {

    private static NKLandscapes createNK(int n, int k, long seed) {
        NKLandscapes pbf = new NKLandscapes();
        Properties prop = new Properties();
        prop.setProperty(NKLandscapes.N_STRING, "" + n);
        prop.setProperty(NKLandscapes.K_STRING, "" + k);
        prop.setProperty(NKLandscapes.CIRCULAR_STRING, "random");
        pbf.setConfiguration(prop);
        pbf.setSeed(seed);
        return pbf;
    }

    private static MultiObjectiveLandscape createBiObjective(int n, int k, long seed) {
        EmbeddedLandscape[] lands = new EmbeddedLandscape[]{
                createNK(n, k, seed),
                createNK(n, k, seed + 9999L)
        };
        MultiObjectiveLandscape mo = new MultiObjectiveLandscape(lands);
        mo.setWeights(new double[]{0.5, 0.5});
        return mo;
    }

    /**
     * Calculates 2D Hypervolume for a non-dominated set relative to a nadir reference point.
     * Assumes maximization.
     */
    private static double computeHypervolume2D(List<double[]> points, double ref1, double ref2) {
        if (points.isEmpty()) return 0.0;

        // Sort descending by objective 1
        List<double[]> sorted = new ArrayList<>(points);
        sorted.sort((a, b) -> Double.compare(b[0], a[0]));

        double hv = 0.0;
        double currentMaxF2 = ref2;

        for (double[] pt : sorted) {
            double f1 = pt[0];
            double f2 = pt[1];

            if (f1 > ref1 && f2 > currentMaxF2) {
                hv += (f1 - ref1) * (f2 - currentMaxF2);
                currentMaxF2 = f2;
            }
        }
        return hv;
    }

    /**
     * Checks if solution 'a' strictly dominates solution 'b' (all >= and at least one >).
     */
    private static boolean dominates(double[] a, double[] b) {
        boolean strictlyBetter = false;
        for (int i = 0; i < a.length; i++) {
            if (a[i] < b[i]) return false;
            if (a[i] > b[i]) strictlyBetter = true;
        }
        return strictlyBetter;
    }

    public static void runQualityComparison(int n, int k, int numTrials) {
        System.out.println("==========================================================================");
        System.out.printf("PARETO QUALITY BENCHMARK: N=%d, K=%d, Trials=%d\n", n, k, numTrials);
        System.out.println("==========================================================================");

        MultiObjectiveLandscape mo = createBiObjective(n, k, 777L);

        List<PBSolution> blueParents = new ArrayList<>(numTrials);
        List<PBSolution> redParents = new ArrayList<>(numTrials);
        for (int i = 0; i < numTrials; i++) {
            blueParents.add(mo.getRandomSolution());
            redParents.add(mo.getRandomSolution());
        }

        NonDominatedSet qpxArchive = new NonDominatedSet(n);
        QPartitionCrossover qpx = new QPartitionCrossover(mo);
        qpx.setArchive(qpxArchive);

        NonDominatedSet mopxArchive = new NonDominatedSet(n);
        TransitionQPartitionCrossover mopx = new TransitionQPartitionCrossover(mo);
        mopx.setArchive(mopxArchive);
        mopx.setDelta(0.0); // Full transition sweep

        NonDominatedSet mopxWindowArchive = new NonDominatedSet(n);
        TransitionQPartitionCrossover mopxWindow = new TransitionQPartitionCrossover(mo);
        mopxWindow.setArchive(mopxWindowArchive);
        mopxWindow.setDelta(0.05); // Focused micro-sweep

        // Run across multiple weights along the simplex sweep (like McDRILS-MO)
        double[] weightSteps = {0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.6, 0.7, 0.8, 0.9, 0.95};
        int trialsPerWeight = Math.max(5, numTrials / weightSteps.length);

        for (double w1 : weightSteps) {
            double w2 = 1.0 - w1;
            mo.setWeights(new double[]{w1, w2});

            for (int t = 0; t < trialsPerWeight; t++) {
                PBSolution b = mo.getRandomSolution();
                PBSolution r = mo.getRandomSolution();

                qpx.recombineInternal(b, r);
                mopx.recombineInternal(b, r);
                mopxWindow.recombineInternal(b, r);
            }
        }

        List<double[]> qpxSols = qpxArchive.getSolutions();
        List<double[]> mopxSols = mopxArchive.getSolutions();
        List<double[]> mopxWinSols = mopxWindowArchive.getSolutions();

        // 1. Min/Max Ranges (Spread)
        double minF1_qpx = Double.POSITIVE_INFINITY, maxF1_qpx = Double.NEGATIVE_INFINITY;
        double minF2_qpx = Double.POSITIVE_INFINITY, maxF2_qpx = Double.NEGATIVE_INFINITY;
        for (double[] s : qpxSols) {
            minF1_qpx = Math.min(minF1_qpx, s[0]);
            maxF1_qpx = Math.max(maxF1_qpx, s[0]);
            minF2_qpx = Math.min(minF2_qpx, s[1]);
            maxF2_qpx = Math.max(maxF2_qpx, s[1]);
        }

        double minF1_mopx = Double.POSITIVE_INFINITY, maxF1_mopx = Double.NEGATIVE_INFINITY;
        double minF2_mopx = Double.POSITIVE_INFINITY, maxF2_mopx = Double.NEGATIVE_INFINITY;
        for (double[] s : mopxSols) {
            minF1_mopx = Math.min(minF1_mopx, s[0]);
            maxF1_mopx = Math.max(maxF1_mopx, s[0]);
            minF2_mopx = Math.min(minF2_mopx, s[1]);
            maxF2_mopx = Math.max(maxF2_mopx, s[1]);
        }

        // Global reference point for Hypervolume
        double refF1 = Math.min(minF1_qpx, minF1_mopx) - 1.0;
        double refF2 = Math.min(minF2_qpx, minF2_mopx) - 1.0;

        double hvQpx = computeHypervolume2D(qpxSols, refF1, refF2);
        double hvMopx = computeHypervolume2D(mopxSols, refF1, refF2);
        double hvMopxWin = computeHypervolume2D(mopxWinSols, refF1, refF2);

        // 2. Mutual Dominance: How many QPX solutions are dominated by at least one MOPX solution?
        int qpxDominatedByMopx = 0;
        for (double[] q : qpxSols) {
            for (double[] m : mopxSols) {
                if (dominates(m, q)) {
                    qpxDominatedByMopx++;
                    break;
                }
            }
        }

        int mopxDominatedByQpx = 0;
        for (double[] m : mopxSols) {
            for (double[] q : qpxSols) {
                if (dominates(q, m)) {
                    mopxDominatedByQpx++;
                    break;
                }
            }
        }

        System.out.printf("%-28s | %-12s | %-14s | %-14s | %-16s\n", "Operator", "Archive Size", "Range f1", "Range f2", "Hypervolume");
        System.out.println("-----------------------------+--------------+----------------+----------------+-----------------");
        System.out.printf("%-28s | %10d   | [%6.1f, %6.1f] | [%6.1f, %6.1f] | %15.2f\n",
                "QPX (Hamming-1 Neighbors)", qpxSols.size(), minF1_qpx, maxF1_qpx, minF2_qpx, maxF2_qpx, hvQpx);
        System.out.printf("%-28s | %10d   | [%6.1f, %6.1f] | [%6.1f, %6.1f] | %15.2f\n",
                "MOPX (Full Transition Sweep)", mopxSols.size(), minF1_mopx, maxF1_mopx, minF2_mopx, maxF2_mopx, hvMopx);
        System.out.printf("%-28s | %10d   | %14s | %14s | %15.2f\n",
                "MOPX (Micro-Sweep d=0.05)", mopxWinSols.size(), "--", "--", hvMopxWin);
        System.out.println("--------------------------------------------------------------------------");
        System.out.println("MUTUAL DOMINANCE ANALYSIS:");
        System.out.printf("  * QPX solutions strictly dominated by MOPX:  %d / %d (%.1f%%)\n",
                qpxDominatedByMopx, qpxSols.size(), (qpxSols.size() > 0 ? 100.0 * qpxDominatedByMopx / qpxSols.size() : 0.0));
        System.out.printf("  * MOPX solutions strictly dominated by QPX:  %d / %d (%.1f%%)\n",
                mopxDominatedByQpx, mopxSols.size(), (mopxSols.size() > 0 ? 100.0 * mopxDominatedByQpx / mopxSols.size() : 0.0));
        System.out.printf("  * Hypervolume Gain (MOPX vs QPX):            %+.2f%%\n",
                (hvQpx > 0 ? ((hvMopx - hvQpx) / hvQpx) * 100.0 : 0.0));
        System.out.println();
    }

    public static void runSingleCrossoverSpreadAnalysis() {
        System.out.println("==========================================================================");
        System.out.println("SINGLE CROSSOVER EVENT ANALYSIS (N=1000, K=2, Components q)");
        System.out.println("Verifying Darrell Whitley's Hypothesis on Offspring Clustering vs Spread");
        System.out.println("==========================================================================");

        int n = 1000;
        int k = 2;
        MultiObjectiveLandscape mo = createBiObjective(n, k, 12345L);
        mo.setWeights(new double[]{0.5, 0.5});

        PBSolution blue = mo.getRandomSolution();
        PBSolution red = mo.getRandomSolution();

        NonDominatedSet qpxArch = new NonDominatedSet(n);
        QPartitionCrossover qpx = new QPartitionCrossover(mo);
        qpx.setArchive(qpxArch);
        qpx.recombineInternal(blue, red);

        NonDominatedSet mopxArch = new NonDominatedSet(n);
        TransitionQPartitionCrossover mopx = new TransitionQPartitionCrossover(mo);
        mopx.setArchive(mopxArch);
        mopx.recombineInternal(blue, red);

        System.out.println("Decomposed components q = " + mopx.getNumberOfComponents());
        System.out.println("QPX  harvested into archive: " + qpxArch.getSolutions().size() + " non-dominated solutions");
        System.out.println("MOPX harvested into archive: " + mopxArch.getSolutions().size() + " non-dominated solutions");

        // Inspect MOPX harvested solutions
        System.out.println("\nMOPX Harvested Objective Points:");
        for (double[] pt : mopxArch.getSolutions()) {
            System.out.printf("  f1 = %8.2f,  f2 = %8.2f\n", pt[0], pt[1]);
        }

        System.out.println("\nQPX Harvested Objective Points:");
        for (double[] pt : qpxArch.getSolutions()) {
            System.out.printf("  f1 = %8.2f,  f2 = %8.2f\n", pt[0], pt[1]);
        }
        System.out.println();
    }

    public static void main(String[] args) {
        System.out.println("Starting Pareto Quality & Solution Coverage Benchmark\n");

        runQualityComparison(100, 2, 200);
        runQualityComparison(500, 2, 200);
        runQualityComparison(1000, 2, 100);

        runSingleCrossoverSpreadAnalysis();

        System.out.println("QUALITY BENCHMARK COMPLETED.");
    }
}
