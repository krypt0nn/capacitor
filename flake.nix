{
    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
        flake-utils.url = "github:numtide/flake-utils";

        rust-overlay = {
            url = "github:oxalica/rust-overlay";
            inputs.nixpkgs.follows = "nixpkgs";
        };
    };

    outputs = { self, nixpkgs, flake-utils, rust-overlay }:
        flake-utils.lib.eachDefaultSystem (system:
            let
                pkgs = import nixpkgs {
                    inherit system;

                    overlays = [ rust-overlay.overlays.default ];
                };

                muslPkgs = pkgs.pkgsCross.musl64;

                config = pkgs.lib.importTOML ./Cargo.toml;
            in {
                packages.default = muslPkgs.rustPlatform.buildRustPackage {
                    pname = config.package.name;
                    version = config.package.version;

                    src = ./.;
                    cargoLock.lockFile = ./Cargo.lock;

                    doCheck = false;
                };

                devShells.default = pkgs.mkShell {
                    nativeBuildInputs = with pkgs; [
                        (pkgs.pkgs.rust-bin.stable.latest.default.override {
                            targets = [ "x86_64-unknown-linux-musl" ];
                            extensions = [ "rust-src" ];
                        })
                        muslPkgs.buildPackages.gcc
                    ];
                };
            });
}
