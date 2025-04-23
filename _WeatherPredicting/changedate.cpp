#include <iostream>
#include <cmath>

#define PI 3.14159265358979323846

using namespace std;

// Kiểm tra năm nhuận
bool isLeapYear(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

// Tính số ngày từ đầu năm
int daysFromStart(int day, int month, int year) {
    int daysInMonth[] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    if (isLeapYear(year)) {
        daysInMonth[1] = 29; // Tháng 2 có 29 ngày nếu là năm nhuận
    }
    int totalDays = day;
    for (int i = 0; i < month - 1; i++) {
        totalDays += daysInMonth[i];
    }
    return totalDays;
}

// Mã hóa tuần hoàn bằng sin/cos
void encodeSinCos(int day, int month, int year, double &encodedValue) {
    int dayOfYear = daysFromStart(day, month, year);
    int totalDaysInYear = isLeapYear(year) ? 366 : 365;
    encodedValue = sin(2 * PI * dayOfYear / totalDaysInYear) + cos(2 * PI * dayOfYear / totalDaysInYear);
}

int main() {
    freopen("date.txt", "r", stdin);   // Đọc từ file date.txt
    freopen("datewanted", "w", stdout); // Ghi output vào file datewanted

    int day, month, year;
    char slash1, slash2; // Dùng để nhận ký tự '/'

    // Đọc liên tục đến khi hết file
    while (cin >> day >> slash1 >> month >> slash2 >> year) {
        // Kiểm tra định dạng
        if (slash1 != '/' || slash2 != '/') {
            cerr << "Sai dinh dang ngay/thang/nam!" << endl;
            continue; // Bỏ qua dòng lỗi và đọc dòng tiếp theo
        }

        double encodedValue;
        encodeSinCos(day, month, year, encodedValue);

        cout << encodedValue << endl;
    }

    return 0;
}
