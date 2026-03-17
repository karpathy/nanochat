// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "NanochatMLXStub",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "nanochat-mlx-stub", targets: ["NanochatMLXStub"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.6"),
    ],
    targets: [
        .executableTarget(
            name: "NanochatMLXStub",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        )
    ]
)