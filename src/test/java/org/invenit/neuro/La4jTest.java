package org.invenit.neuro;

import org.la4j.Matrix;
import org.la4j.Vector;
import org.la4j.matrix.dense.Basic1DMatrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.vector.dense.BasicVector;
import org.testng.annotations.Test;

/**
 * @author Vycheslav Mischeryakov (vmischeryakov@gmail.com)
 */
public class La4jTest {

    @Test
    public void testMatrixVectorMult() {
        Matrix matrix = new Basic2DMatrix(new double[][] {
            {1, 2, 3},
            {4, 5, 6}
        });
        Vector vector = new BasicVector(new double[] {1, 2, 3});

        Vector result = matrix.multiply(vector);
        System.out.println(result);
    }

    @Test
    public void testVectorProduct() {
        Vector v1 = new BasicVector(new double[] {1, 2, 3});
        Vector v2 = new BasicVector(new double[] {1, 2, 3});

        System.out.println(v1.hadamardProduct(v2));
    }
}
