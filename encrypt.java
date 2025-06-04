import java.io.File;
import java.util.Scanner;

public class encrypt {
	
	
	public static String encrypt(String plainString, int[] matrix, int matrixD) { //Does the Hill Cipher
		

		char[] TempMessageArray = new char[10000];
		int[] MessageArrayNum = new int[plainString.length()];
		int[] encryptedArrayNum = new int[plainString.length()];
		String encryptedString = "";
		int mIndex = 0;
		int pIndex = 0;
		int eIndex = 0;
		
		for(int i = 0; i < plainString.length(); i++) { //Transfers plain text into char array
			TempMessageArray[i] = plainString.charAt(i);
		}
		
		for(int i = 0; i < plainString.length(); i++) {
			MessageArrayNum[i] = Character.getNumericValue(TempMessageArray[i])-10; //Now 0-25, 0 = a, 25 = z
		}
		
		

		
		
		for (int i = 0; i < plainString.length(); i += matrixD) {
			for (int row = 0; row < matrixD; row++) {
				int sum = 0;
				for (int col = 0; col < matrixD; col++) {
					sum += matrix[row * matrixD + col] * MessageArrayNum[i + col];
				}
				encryptedArrayNum[i + row] = sum % 26; // mod 26 here directly
			}
		}
			

		
		
		for(int i = 0; i < plainString.length(); i++) {
			encryptedArrayNum[i] = encryptedArrayNum[i]%26; //Mod by 26 since alphabet is 26
		}
		
		for(int i = 0; i < plainString.length(); i++) {
			encryptedString = encryptedString + (char)(encryptedArrayNum[i]+97); //Reverts back to alphabet
		}
		
		return encryptedString;
	}
	
	public static void printMatrix(int size, int[] matrixArray) { //Prints the Key Matrix
		int temp = 0;
		System.out.println();
		System.out.println("Key matrix:");
		
		for(int i = 0; i < size; i++) {

			
			for(int j = 0; j < size; j++) {
				if(String.valueOf(matrixArray[temp]).length() == 1) {
					System.out.print("   ");
				} else if(String.valueOf(matrixArray[temp]).length() == 2) {
					System.out.print("  ");
				} else if(String.valueOf(matrixArray[temp]).length() == 3) {
					System.out.print(" ");
				}
				System.out.print(matrixArray[temp]);

				temp++;
			}
			System.out.println();
		}
		System.out.println();
	}
	
	public static void printText(String plainString, String encryptString) { //Prints out the strings
		System.out.println(plainString);
		System.out.println("Plaintext:");
		int z = 0;
		
		int x = (int)Math.ceil(plainString.length() / 80.0);

		String[] temp = new String[x];
		
		for(int i = 0; i < x; i++) {
			temp[i] = "";
		}
		

		for(int i = 0; i < x; i++) {
			for(int j = 0; j < 80; j++) {
				temp[i] = temp[i] + plainString.charAt(z);
				z++;
				if(z >= plainString.length()) {
					break;
				}
			}
		}
		

		
		for(int i = 0; i < x; i++) {
			System.out.println(temp[i]);
		}

		z = 0;
		
		
		System.out.println("Ciphertext:");
		
		
		
		String[] temp1 = new String[x];
		for(int i = 0; i < x; i++) {
			temp1[i] = "";
		}
		for(int i = 0; i < x; i++) {
			for(int j = 0; j < 80; j++) {
				temp1[i] = temp1[i] + encryptString.charAt(z);
				z++;
				if(z >= plainString.length()) {
					break;
				}
			}
		}
		
		//for(int i = 0; i < x; i++) {
		//	temp1[i] = temp1[i].replaceAll("null","");
		//}
		
		for(int i = 0; i < x; i++) {
			System.out.println(temp1[i]);
		}

	}
	
	public static String removeFiller(String string) { //Cleans the string

		string = string.toLowerCase();
		string = string.replaceAll("[^a-z]+","");
		string = string.replaceAll("null", "");
		return string;
	}

	public static void main(String[] args) throws Exception {
		
		int[] matrixArray = new int[100];
		
		//Scanners to scan both files
		File keyFile = new File(args[0]);
		Scanner keyScan = new Scanner(keyFile);
		
		File textFile = new File(args[1]);
		Scanner textScan = new Scanner(textFile);
		
		//Scans first int of key file
		int matrix = Integer.parseInt(keyScan.nextLine());
		
		//Scans the rest of key
		int temp = 0;
		while(keyScan.hasNext()) {
			matrixArray[temp] = Integer.parseInt(keyScan.next());
			temp++;
		}
		
		printMatrix(matrix, matrixArray);
		
		//Scans the plain text
		temp = 0;
		String tempString = "";
		while(textScan.hasNext()) {
			tempString = tempString + textScan.nextLine();
		}
		
		tempString = removeFiller(tempString); //Makes it into simple string and puts X if needed
		
		int remainder = tempString.length() % matrix;
		if (remainder != 0) {
			int padding = matrix - remainder;
			for (int i = 0; i < padding; i++) {
				tempString += "x";
			}
		}
		
		String temp1String = tempString;
		
		temp1String = encrypt(tempString, matrixArray, matrix);
		
		printText(tempString, temp1String);
		
	}
	


}