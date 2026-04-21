import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class HamiltonianEstimator {

    // 数据容器
    static class DataPoint {
        double year;
        double x;   // hare
        double y;   // lynx

        DataPoint(double year, double x, double y) {
            this.year = year;
            this.x = x;
            this.y = y;
        }
    }

    static class Params {
        double a, b, c, d;
        double objective;

        Params(double a, double b, double c, double d, double objective) {
            this.a = a;
            this.b = b;
            this.c = c;
            this.d = d;
            this.objective = objective;
        }
    }

    public static void main(String[] args) {
        String filename = "Leigh1968_harelynx.csv";
        List<DataPoint> data = readCSV(filename);

        if (data.isEmpty()) {
            System.out.println("No data loaded.");
            return;
        }

        scaleData(data, 1000.0);

        double avgX = meanX(data);
        double avgY = meanY(data);
        double period = estimatePeriodFromPeaks(data);

        System.out.println("Estimated average hare = " + avgX);
        System.out.println("Estimated average lynx = " + avgY);
        System.out.println("Estimated period T = " + period);

        double aMin = 0.01, aMax = 2.0;
        double bMin = 0.001, bMax = 2.0;
        double cMin = 0.01, cMax = 2.0;
        double dMin = 0.001, dMax = 2.0;

        Params best = randomSearchWithRefinement(
                data, avgX, avgY, period,
                aMin, aMax, bMin, bMax, cMin, cMax, dMin, dMax,
                5,       // refinement rounds
                4000     // samples per round
        );

        System.out.println("\nBest parameters found:");
        System.out.printf("a = %.6f%n", best.a);
        System.out.printf("b = %.6f%n", best.b);
        System.out.printf("c = %.6f%n", best.c);
        System.out.printf("d = %.6f%n", best.d);
        System.out.printf("objective = %.10f%n", best.objective);

        double xStar = best.c / best.d;
        double yStar = best.a / best.b;
        double impliedPeriod = 2.0 * Math.PI / Math.sqrt(best.a * best.c);

        System.out.println("\nDerived quantities:");
        System.out.printf("x* = c/d = %.6f%n", xStar);
        System.out.printf("y* = a/b = %.6f%n", yStar);
        System.out.printf("Approx period = 2pi/sqrt(ac) = %.6f%n", impliedPeriod);

        double[] Hvals = computeHamiltonianValues(data, best.a, best.b, best.c, best.d);
        double hMean = mean(Hvals);
        double hVar = variance(Hvals);

        System.out.println("\nHamiltonian consistency check:");
        System.out.printf("mean(H) = %.8f%n", hMean);
        System.out.printf("var(H)  = %.8f%n", hVar);
    }

    public static List<DataPoint> readCSV(String filename) {
        List<DataPoint> data = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine(); // 跳过表头

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");

                if (parts.length < 3) {
                    continue;
                }

                double year = Double.parseDouble(parts[0].trim());
                double hare = Double.parseDouble(parts[1].trim());
                double lynx = Double.parseDouble(parts[2].trim());

                if (hare > 0 && lynx > 0) {
                    data.add(new DataPoint(year, hare, lynx));
                }
            }
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
        }

        return data;
    }

    public static void scaleData(List<DataPoint> data, double factor) {
        for (DataPoint p : data) {
            p.x /= factor;
            p.y /= factor;
        }
    }

    public static double meanX(List<DataPoint> data) {
        double sum = 0.0;
        for (DataPoint p : data) {
            sum += p.x;
        }
        return sum / data.size();
    }

    public static double meanY(List<DataPoint> data) {
        double sum = 0.0;
        for (DataPoint p : data) {
            sum += p.y;
        }
        return sum / data.size();
    }

    public static double estimatePeriodFromPeaks(List<DataPoint> data) {
        List<Double> peakYears = new ArrayList<>();

        for (int i = 1; i < data.size() - 1; i++) {
            double prev = data.get(i - 1).x;
            double curr = data.get(i).x;
            double next = data.get(i + 1).x;

            if (curr > prev && curr > next) {
                peakYears.add(data.get(i).year);
            }
        }

        if (peakYears.size() < 2) {
            return 10.0;
        }

        double totalGap = 0.0;
        for (int i = 1; i < peakYears.size(); i++) {
            totalGap += (peakYears.get(i) - peakYears.get(i - 1));
        }

        return totalGap / (peakYears.size() - 1);
    }

    public static double hamiltonian(double x, double y, double a, double b, double c, double d) {
        return d * x - c * Math.log(x) + b * y - a * Math.log(y);
    }

    public static double[] computeHamiltonianValues(List<DataPoint> data, double a, double b, double c, double d) {
        double[] vals = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            vals[i] = hamiltonian(data.get(i).x, data.get(i).y, a, b, c, d);
        }
        return vals;
    }

    public static double mean(double[] arr) {
        double sum = 0.0;
        for (double v : arr) {
            sum += v;
        }
        return sum / arr.length;
    }

    public static double variance(double[] arr) {
        double m = mean(arr);
        double sum = 0.0;
        for (double v : arr) {
            double diff = v - m;
            sum += diff * diff;
        }
        return sum / arr.length;
    }

    public static double objective(
            List<DataPoint> data,
            double avgX,
            double avgY,
            double period,
            double a, double b, double c, double d
    ) {
        if (a <= 0 || b <= 0 || c <= 0 || d <= 0) {
            return Double.POSITIVE_INFINITY;
        }

        double[] Hvals = computeHamiltonianValues(data, a, b, c, d);
        double hVar = variance(Hvals);

        double xStar = c / d;
        double yStar = a / b;
        double modelPeriod = 2.0 * Math.PI / Math.sqrt(a * c);

        double eqPenaltyX = square(xStar - avgX);
        double eqPenaltyY = square(yStar - avgY);
        double periodPenalty = square(modelPeriod - period);

    
        double w1 = 1.0;
        double w2 = 5.0;
        double w3 = 5.0;
        double w4 = 2.0;

        return w1 * hVar
                + w2 * eqPenaltyX
                + w3 * eqPenaltyY
                + w4 * periodPenalty;
    }

    public static double square(double x) {
        return x * x;
    }

    
    public static Params randomSearchWithRefinement(
            List<DataPoint> data,
            double avgX,
            double avgY,
            double period,
            double aMin, double aMax,
            double bMin, double bMax,
            double cMin, double cMax,
            double dMin, double dMax,
            int rounds,
            int samplesPerRound
    ) {
        Random rand = new Random(42);

        Params best = new Params(0, 0, 0, 0, Double.POSITIVE_INFINITY);

        for (int round = 1; round <= rounds; round++) {
            for (int i = 0; i < samplesPerRound; i++) {
                double a = randomInRange(rand, aMin, aMax);
                double b = randomInRange(rand, bMin, bMax);
                double c = randomInRange(rand, cMin, cMax);
                double d = randomInRange(rand, dMin, dMax);

                double obj = objective(data, avgX, avgY, period, a, b, c, d);

                if (obj < best.objective) {
                    best = new Params(a, b, c, d, obj);
                }
            }

            System.out.println("Round " + round + " best objective = " + best.objective);

            double shrink = 0.35;

            double aWidth = (aMax - aMin) * shrink;
            double bWidth = (bMax - bMin) * shrink;
            double cWidth = (cMax - cMin) * shrink;
            double dWidth = (dMax - dMin) * shrink;

            aMin = Math.max(1e-6, best.a - aWidth / 2.0);
            aMax = best.a + aWidth / 2.0;

            bMin = Math.max(1e-6, best.b - bWidth / 2.0);
            bMax = best.b + bWidth / 2.0;

            cMin = Math.max(1e-6, best.c - cWidth / 2.0);
            cMax = best.c + cWidth / 2.0;

            dMin = Math.max(1e-6, best.d - dWidth / 2.0);
            dMax = best.d + dWidth / 2.0;
        }

        return best;
    }

    public static double randomInRange(Random rand, double min, double max) {
        return min + (max - min) * rand.nextDouble();
    }
}