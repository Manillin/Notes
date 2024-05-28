package SWDesign.designPattern;

public class Main {
    public static void main(String[] args) {
        SingletonClass a = SingletonClass.GetInstance();
        SingletonClass a2 = SingletonClass.GetInstance();
        System.out.println(a.equals(a2));
    }
}