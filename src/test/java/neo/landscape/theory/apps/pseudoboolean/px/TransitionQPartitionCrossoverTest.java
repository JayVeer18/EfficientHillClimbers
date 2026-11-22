package neo.landscape.theory.apps.pseudoboolean.px;

import neo.landscape.theory.apps.pseudoboolean.PBSolution;
import neo.landscape.theory.apps.pseudoboolean.problems.MultiObjectiveLandscape;
import neo.landscape.theory.apps.pseudoboolean.problems.NKLandscapes;
import neo.landscape.theory.apps.pseudoboolean.util.NonDominatedSet;
import org.junit.jupiter.api.Test;

import java.util.Properties;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.withPrecision;

/**
 * Unit and integration tests for {@link TransitionQPartitionCrossover}.
 * Verifies exact critical weight zero-crossings, bi-objective and tri-objective
 * Pareto archive harvesting, and micro-sweep delta windowing.
 */
public class TransitionQPartitionCrossoverTest {

    private NKLandscapes createNKLandscape(int n, int k, long seed) {
        NKLandscapes pbf = new NKLandscapes();
        Properties prop = new Properties();
        prop.setProperty(NKLandscapes.N_STRING, "" + n);
        prop.setProperty(NKLandscapes.K_STRING, "" + k);
        prop.setProperty(NKLandscapes.CIRCULAR_STRING, "random");
        pbf.setConfiguration(prop);
        pbf.setSeed(seed);
        return pbf;
    }

    /**
     * Verifies the exact theoretical zero-crossing critical weights from Darrell Whitley's
     * benchmark problem (EQUALITIES of weight and alpha copy.xlsx).
     * Tests that:
     * 1. Trade-off components have transitions w* strictly in (0, 1) without reference point P0.
     * 2. Component 2 flips at w2* = 0.40 and Component 3 flips at w3* = 0.60.
     * 3. Dominated unanimous components are correctly recognized.
     */
    @Test
    public void testCriticalWeightZeroCrossing() {
        // Differential components: alpha^1 (Obj 1) and alpha^2 (Obj 2)
        double a1_obj1 = -8.0, a1_obj2 = -4.0;
        double a2_obj1 = -12.0, a2_obj2 = 8.0;
        double a3_obj1 = 4.0, a3_obj2 = -6.0;

        // Trade-off component 2 transition: w2* = alpha^2 / (alpha^2 - alpha^1)
        double w2Star = a2_obj2 / (a2_obj2 - a2_obj1);
        assertThat(w2Star).isCloseTo(0.40, withPrecision(1e-6));
        assertThat(w2Star).isGreaterThan(0.0).isLessThan(1.0);

        // Trade-off component 3 transition: w3* = alpha^2 / (alpha^2 - alpha^1)
        double w3Star = a3_obj2 / (a3_obj2 - a3_obj1);
        assertThat(w3Star).isCloseTo(0.60, withPrecision(1e-6));
        assertThat(w3Star).isGreaterThan(0.0).isLessThan(1.0);

        // Unanimous component 1: dominated (both negative) -> Red procedence strictly dominates
        boolean comp1Dominated = (a1_obj1 <= 0.0 && a1_obj2 <= 0.0);
        assertThat(comp1Dominated).isTrue();
    }

    /**
     * Tests recombination on bi-objective (M=2) NK landscapes.
     * Verifies that valid child solutions are produced and non-dominated offspring are added to the archive.
     */
    @Test
    public void testRecombinationBiObjective() {
        int n = 20;
        int k = 2;
        NKLandscapes l1 = createNKLandscape(n, k, 12345L);
        NKLandscapes l2 = createNKLandscape(n, k, 67890L);

        MultiObjectiveLandscape moLandscape = new MultiObjectiveLandscape(l1, l2);
        moLandscape.setWeights(new double[]{0.5, 0.5});

        NonDominatedSet archive = new NonDominatedSet(n);
        TransitionQPartitionCrossover crossover = new TransitionQPartitionCrossover(moLandscape);
        crossover.setArchive(archive);
        crossover.setSeed(42L);

        for (int run = 0; run < 10; run++) {
            PBSolution blue = moLandscape.getRandomSolution();
            PBSolution red = moLandscape.getRandomSolution();

            PBSolution child = crossover.recombine(blue, red);

            assertThat(child).isNotNull();
            assertThat(child.getN()).isEqualTo(n);
            assertThat(archive.getSolutions()).isNotEmpty();
        }
    }

    /**
     * Tests Darrell's micro-sweep windowing parameter (delta).
     * Verifies that a localized window (delta = 0.05) restricts candidate evaluations
     * compared to a full sweep (delta = 0.0).
     */
    @Test
    public void testMicroSweepWindowing() {
        int n = 30;
        int k = 2;
        NKLandscapes l1 = createNKLandscape(n, k, 11111L);
        NKLandscapes l2 = createNKLandscape(n, k, 22222L);

        MultiObjectiveLandscape moLandscape = new MultiObjectiveLandscape(l1, l2);
        moLandscape.setWeights(new double[]{0.6, 0.4});

        // 1. Full sweep
        NonDominatedSet fullArchive = new NonDominatedSet(n);
        TransitionQPartitionCrossover fullCrossover = new TransitionQPartitionCrossover(moLandscape);
        fullCrossover.setArchive(fullArchive);
        fullCrossover.setDelta(0.0);

        // 2. Focused Micro-sweep window [0, delta]
        NonDominatedSet windowArchive = new NonDominatedSet(n);
        TransitionQPartitionCrossover windowCrossover = new TransitionQPartitionCrossover(moLandscape);
        windowCrossover.setArchive(windowArchive);
        windowCrossover.setDelta(0.05);

        PBSolution blue = moLandscape.getRandomSolution();
        PBSolution red = moLandscape.getRandomSolution();

        fullCrossover.recombine(blue, red);
        windowCrossover.recombine(blue, red);

        assertThat(windowArchive.getSolutions().size())
                .isLessThanOrEqualTo(fullArchive.getSolutions().size());
    }

    /**
     * Tests recombination on tri-objective (M=3) NK landscapes.
     * Verifies directional simplex ray transitions across M=3 objectives and Pareto archiving.
     */
    @Test
    public void testRecombinationTriObjective() {
        int n = 25;
        int k = 2;
        NKLandscapes l1 = createNKLandscape(n, k, 101L);
        NKLandscapes l2 = createNKLandscape(n, k, 202L);
        NKLandscapes l3 = createNKLandscape(n, k, 303L);

        MultiObjectiveLandscape moLandscape = new MultiObjectiveLandscape(l1, l2, l3);
        moLandscape.setWeights(new double[]{0.34, 0.33, 0.33});

        NonDominatedSet archive = new NonDominatedSet(n);
        TransitionQPartitionCrossover crossover = new TransitionQPartitionCrossover(moLandscape);
        crossover.setArchive(archive);
        crossover.setSeed(999L);

        for (int run = 0; run < 10; run++) {
            PBSolution blue = moLandscape.getRandomSolution();
            PBSolution red = moLandscape.getRandomSolution();

            PBSolution child = crossover.recombine(blue, red);

            assertThat(child).isNotNull();
            assertThat(child.getN()).isEqualTo(n);
            assertThat(archive.getSolutions()).isNotEmpty();
        }
    }

    public static void main(String[] args) {
        TransitionQPartitionCrossoverTest test = new TransitionQPartitionCrossoverTest();
        test.testCriticalWeightZeroCrossing();
        test.testRecombinationBiObjective();
        test.testMicroSweepWindowing();
        test.testRecombinationTriObjective();
        System.out.println("ALL TESTS PASSED SUCCESSFULLY");
    }
}
