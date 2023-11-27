double performOperations(double input) {
    // 计算校验和
    double checksum = calculateChecksum(input);

    // 将校验和乘以 773493838/1114245777
    checksum = checksum * 773493838 / 1114245777;

    // 除以 1e+4
    checksum = checksum / 1e+4;

    // 减去 78000000000
    checksum = checksum - 78000000000;

    return checksum;
}
