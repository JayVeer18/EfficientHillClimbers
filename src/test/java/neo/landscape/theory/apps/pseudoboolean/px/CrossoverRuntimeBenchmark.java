package neo.landscape.theory.apps.pseudoboolean.px;

import neo.landscape.theory.apps.pseudoboolean.PBSolution;
import neo.landscape.theory.apps.pseudoboolean.problems.EmbeddedLandscape;
import neo.landscape.theory.apps.pseudoboolean.problems.MultiObjectiveLandscape;
import neo.landscape.theory.apps.pseudoboolean.problems.NKLandscapes;
import neo.landscape.theory.apps.pseudoboolean.util.NonDominatedSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Direct runtime and archive quality benchmarking comparing QPX vs. MOPX (Transition-QPX).
 * Measures per-crossover runtime in microseconds and non-dominated archive sizes across identical parent pairs.
 */
public class CrossoverRuntimeBenchmark {

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

    private static MultiObjectiveLandscape createMoLandscape(int n, int k, int mObjs, long baseSeed) {
        EmbeddedLandscape[] lands = new EmbeddedLandscape[mObjs];
        double[] weights = new double[mObjs];
        for (int i = 0; i < mObjs; i++) {
            lands[i] = createNK(n, k, baseSeed + i * 1000L);
            weights[i] = 1.0 / mObjs;
        }
        MultiObjectiveLandscape mo = new MultiObjectiveLandscape(lands);
        mo.setWeights(weights);
        return mo;
    }

    public static void runBenchmark(int n, int k, int mObjs, int numTrials) {
        System.out.println("==========================================================================");
        System.out.printf("BENCHMARK CONFIG: N=%d, K=%d, Objectives M=%d, Trials=%d\n", n, k, mObjs, numTrials);
        System.out.println("==========================================================================");

        MultiObjectiveLandscape mo = createMoLandscape(n, k, mObjs, 42L);

        // Pre-generate identical parent pairs so all crossovers evaluate identical problems
        List<PBSolution> blueParents = new ArrayList<>(numTrials);
        List<PBSolution> redParents = new ArrayList<>(numTrials);
        for (int i = 0; i < numTrials; i++) {
            blueParents.add(mo.getRandomSolution());
            redParents.add(mo.getRandomSolution());
        }

        // 1. Warm-up JIT
        NonDominatedSet warmupArchive = new NonDominatedSet(n);
        QPartitionCrossover warmupQpx = new QPartitionCrossover(mo);
        warmupQpx.setArchive(warmupArchive);
        TransitionQPartitionCrossover warmupMopx = new TransitionQPartitionCrossover(mo);
        warmupMopx.setArchive(warmupArchive);
        for (int i = 0; i < 30; i++) {
            warmupQpx.recombineInternal(blueParents.get(i % numTrials), redParents.get(i % numTrials));
            warmupMopx.recombineInternal(blueParents.get(i % numTrials), redParents.get(i % numTrials));
        }

        // --- Benchmark QPX ---
        NonDominatedSet qpxArchive = new NonDominatedSet(n);
        QPartitionCrossover qpx = new QPartitionCrossover(mo);
        qpx.setArchive(qpxArchive);
        qpx.setSeed(100L);

        long qpxStart = System.nanoTime();
        for (int i = 0; i < numTrials; i++) {
            qpx.recombineInternal(blueParents.get(i), redParents.get(i));
        }
        long qpxDurationNs = System.nanoTime() - qpxStart;
        double qpxTimeMs = qpxDurationNs / 1_000_000.0;
        double qpxPerCrossoverUs = (qpxDurationNs / (double) numTrials) / 1_000.0;

        // --- Benchmark MOPX (Full Transition Sweep: delta=0.0) ---
        NonDominatedSet mopxFullArchive = new NonDominatedSet(n);
        TransitionQPartitionCrossover mopxFull = new TransitionQPartitionCrossover(mo);
        mopxFull.setArchive(mopxFullArchive);
        mopxFull.setDelta(0.0);
        mopxFull.setSeed(100L);

        long mopxFullStart = System.nanoTime();
        for (int i = 0; i < numTrials; i++) {
            mopxFull.recombineInternal(blueParents.get(i), redParents.get(i));
        }
        long mopxFullDurationNs = System.nanoTime() - mopxFullStart;
        double mopxFullTimeMs = mopxFullDurationNs / 1_000_000.0;
        double mopxFullPerCrossoverUs = (mopxFullDurationNs / (double) numTrials) / 1_000.0;

        // --- Benchmark MOPX (Micro-Sweep Window: delta=0.05) ---
        NonDominatedSet mopxWindowArchive = new NonDominatedSet(n);
        TransitionQPartitionCrossover mopxWindow = new TransitionQPartitionCrossover(mo);
        mopxWindow.setArchive(mopxWindowArchive);
        mopxWindow.setDelta(0.05);
        mopxWindow.setSeed(100L);

        long mopxWindowStart = System.nanoTime();
        for (int i = 0; i < numTrials; i++) {
            mopxWindow.recombineInternal(blueParents.get(i), redParents.get(i));
        }
        long mopxWindowDurationNs = System.nanoTime() - mopxWindowStart;
        double mopxWindowTimeMs = mopxWindowDurationNs / 1_000_000.0;
        double mopxWindowPerCrossoverUs = (mopxWindowDurationNs / (double) numTrials) / 1_000.0;

        // Print Results Table
        System.out.printf("%-26s | %-14s | %-18s | %-16s\n", "Operator", "Total Time", "Per Crossover", "Archive Size");
        System.out.println("---------------------------+----------------+--------------------+-----------------");
        System.out.printf("%-26s | %10.2f ms | %14.2f us | %12d sols\n", "QPX (Best + Neighbors)", qpxTimeMs, qpxPerCrossoverUs, qpxArchive.getSolutions().size());
        System.out.printf("%-26s | %10.2f ms | %14.2f us | %12d sols\n", "MOPX (Full Sweep)", mopxFullTimeMs, mopxFullPerCrossoverUs, mopxFullArchive.getSolutions().size());
        System.out.printf("%-26s | %10.2f ms | %14.2f us | %12d sols\n", "MOPX (Micro-Sweep d=0.05)", mopxWindowTimeMs, mopxWindowPerCrossoverUs, mopxWindowArchive.getSolutions().size());
        System.out.println();
    }

    public static void main(String[] args) {
        System.out.println("Starting Runtime & Quality Benchmark: QPX vs. MOPX\n");

        // Scenario 1: Moderate size N=100, M=2 (bi-objective)
        runBenchmark(100, 2, 2, 200);

        // Scenario 2: Large scale N=500, M=2 (bi-objective)
        runBenchmark(500, 2, 2, 200);

        // Scenario 3: Large scale N=1000, M=2 (bi-objective)
        runBenchmark(1000, 2, 2, 100);

        // Scenario 4: Tri-objective N=500, M=3
        runBenchmark(500, 2, 3, 100);

        System.out.println("BENCHMARK COMPLETED SUCCESSFULLY.");
    }
}
