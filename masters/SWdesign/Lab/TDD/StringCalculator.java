import java.util.ArrayList;

public class StringCalculator {

    public static int Add(String numbers) {
        int sum = 0;
        if (numbers.isEmpty()) {
            return sum;
        }
        int n1;
        int n2;
        int delimiterIndex = numbers.indexOf(',');

        if (delimiterIndex == -1) {
            n1 = Integer.parseInt(numbers);
            sum += n1;
            return sum;
        }

        n1 = Integer.parseInt(numbers.substring(0, delimiterIndex));
        n2 = Integer.parseInt(numbers.substring(delimiterIndex + 1, numbers.length() - 1));

        sum = n1 + n2;
        return sum;
    }

    public static int Add2(String numbers) {
        int sum = 0;
        ArrayList<Integer> int_numbers = new ArrayList<Integer>();
        char string_elements[] = numbers.toCharArray();
        char delimiter = ',';
        String string_number = "";

        for (int i = 0; i < numbers.length() - 1; i++) {
            if (string_elements[i] == delimiter || string_elements[i] == '\\') {

                if (string_elements[i] == '\\') {
                    i += 1;
                }
                if (string_number == "") {
                    int_numbers.add(0);
                } else {
                    int_numbers.add(Integer.parseInt(string_number));
                    string_number = "";
                }
            } else {
                string_number += string_elements[i];
            }
        }
        for (int i = 0; i < int_numbers.size() - 1; i++) {
            sum += int_numbers.get(i);
        }
        return sum;
    }
}