import java.io.File;
import java.util.Scanner;

public class decrypt {

    public static int modInverse(int a, int m) { //Finds modular inverse
        a = a % m;
        for (int x = 1; x < m; x++) {
            if ((a * x) % m == 1) return x;
        }
        return -1;
    }

    public static int[][] getCofactor(int[][] matrix, int p, int q, int n) { //Gets cofactor matrix
        int[][] temp = new int[n - 1][n - 1];
        int i = 0, j = 0;

        for (int row = 0; row < n; row++) {
            for (int col = 0; col < n; col++) {
                if (row != p && col != q) {
                    temp[i][j++] = matrix[row][col];
                    if (j == n - 1) {
                        j = 0;
                        i++;
                    }
                }
            }
        }

        return temp;
    }

    public static int determinant(int[][] matrix, int n) { //Finds determinant of matrix
        if (n == 1) return matrix[0][0];

        int D = 0;
        int sign = 1;

        for (int f = 0; f < n; f++) {
            int[][] temp = getCofactor(matrix, 0, f, n);
            D += sign * matrix[0][f] * determinant(temp, n - 1);
            sign = -sign;
        }

        return D;
    }

    public static int[][] adjoint(int[][] matrix, int n) { //Finds adjoint of matrix
        int[][] adj = new int[n][n];

        if (n == 1) {
            adj[0][0] = 1;
            return adj;
        }

        int sign;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int[][] temp = getCofactor(matrix, i, j, n);
                sign = ((i + j) % 2 == 0) ? 1 : -1;
                adj[j][i] = (sign * determinant(temp, n - 1)) % 26;
                if (adj[j][i] < 0) adj[j][i] += 26;
            }
        }

        return adj;
    }

    public static int[] inverseMatrix(int[] flatMatrix, int n) { //Finds inverse of matrix
        int[][] matrix = new int[n][n];
        for (int i = 0; i < n * n; i++) {
            matrix[i / n][i % n] = flatMatrix[i];
        }

        int det = determinant(matrix, n) % 26;
        if (det < 0) det += 26;

        int detInv = modInverse(det, 26);

        int[][] adj = adjoint(matrix, n);
        int[] inverseFlat = new int[n * n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverseFlat[i * n + j] = (adj[i][j] * detInv) % 26;
                if (inverseFlat[i * n + j] < 0) inverseFlat[i * n + j] += 26;
            }
        }

        return inverseFlat;
    }

    public static String decrypter(String cipherString, int[] matrix, int matrixD) { //Does the Hill Cipher Decryption
        int[] cipherNums = new int[cipherString.length()];
        int[] plainNums = new int[cipherString.length()];
        StringBuilder decrypted = new StringBuilder();

        for (int i = 0; i < cipherString.length(); i++) { //Transfers cipher text to int array
            cipherNums[i] = cipherString.charAt(i) - 'a';
        }

        int[] inverseMatrix = inverseMatrix(matrix, matrixD); //Gets inverse matrix

        for (int i = 0; i < cipherString.length(); i += matrixD) {
            for (int row = 0; row < matrixD; row++) {
                int sum = 0;
                for (int col = 0; col < matrixD; col++) {
                    sum += inverseMatrix[row * matrixD + col] * cipherNums[i + col];
                }
                plainNums[i + row] = (sum % 26 + 26) % 26;
            }
        }

        for (int num : plainNums) { //Converts back to characters
            decrypted.append((char) (num + 'a'));
        }

        return decrypted.toString();
    }

    public static void printMatrix(int size, int[] matrixArray) { //Prints the Key Matrix
        System.out.println("\nKey matrix:");
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int value = matrixArray[i * size + j];
                System.out.printf("%3d", value);
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void printText(String cipherString, String decryptedString) { //Prints the strings
        System.out.println(cipherString);

        System.out.println("Ciphertext:");
        for (int i = 0; i < cipherString.length(); i += 80) {
            System.out.println(cipherString.substring(i, Math.min(i + 80, cipherString.length())));
        }

        System.out.println("Plaintext:");
        for (int i = 0; i < decryptedString.length(); i += 80) {
            System.out.println(decryptedString.substring(i, Math.min(i + 80, decryptedString.length())));
        }
    }

    public static String cleanInput(String str) { //Cleans the string
        return str.toLowerCase().replaceAll("[^a-z]+", "").replaceAll("null", "");
    }

    public static void main(String[] args) throws Exception { //Main method

        //Scanners to scan both files
        File keyFile = new File(args[0]);
        File cipherFile = new File(args[1]);
        Scanner keyScan = new Scanner(keyFile);
        Scanner cipherScan = new Scanner(cipherFile);

        //Scans first int of key file
        int matrixSize = Integer.parseInt(keyScan.nextLine());
        int[] matrixArray = new int[matrixSize * matrixSize];

        //Scans the rest of key
        for (int i = 0; keyScan.hasNext(); i++) {
            matrixArray[i] = Integer.parseInt(keyScan.next());
        }

        printMatrix(matrixSize, matrixArray);

        //Scans the cipher text
        StringBuilder cipherTextBuilder = new StringBuilder();
        while (cipherScan.hasNextLine()) {
            cipherTextBuilder.append(cipherScan.nextLine());
        }

        String cipherText = cleanInput(cipherTextBuilder.toString()); //Cleans string
        String decryptedText = decrypter(cipherText, matrixArray, matrixSize); //Decrypts string

        printText(cipherText, decryptedText);

        keyScan.close();
        cipherScan.close();
    }
}
