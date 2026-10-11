"""Generate CSP-Wavelet+LOG classification accuracy results."""
if __package__:
    from .result_runner import main
else:
    from result_runner import main

if __name__ == "__main__":
    main("CSP-Wavelet+LOG")
