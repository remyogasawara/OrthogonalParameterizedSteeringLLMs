def run_alpha_zero_sanity_checks_two_class(
    x: torch.Tensor,           # [d] or [n, d]
    target_class: str,
    steering_dirs: dict[str, torch.Tensor],
    target_classes: list[str],
    atol: float = 1e-4,
    verbose: bool = True,
):
    device = x.device
    x_f = x.to(dtype=torch.float32, device=device)
    orig_was_vector = (x_f.dim() == 1)
    x_batch = x_f.unsqueeze(0) if orig_was_vector else x_f  # (n, d)

    # ------------------------------------------------------------------ #
    # Build U_prime, V, V_inv (same as in orthogonalize_vectors_op_two_class)
    # ------------------------------------------------------------------ #
    def _get_vec(name):
        raw = steering_dirs[name][LAYER]["answer_token"]["sample_diff_of_means"]
        return torch.as_tensor(raw, device=device, dtype=torch.float32)

    vectors = [_get_vec(name) for name in target_classes]
    concept_matrix = torch.stack(vectors, dim=1)           # (d, k)

    U_prime, _ = torch.linalg.qr(concept_matrix, mode="reduced")  # (d, k)
    V = U_prime.T @ concept_matrix                                  # (k, k)
    V_inv = torch.linalg.pinv(V)                                    # (k, k)

    # ------------------------------------------------------------------ #
    # Forward decomposition
    # ------------------------------------------------------------------ #
    x_U_coords = x_batch @ U_prime       # (n, k)  -- Step 5
    x_U = x_U_coords @ U_prime.T        # (n, d)  -- span component
    x_null = x_batch - x_U              # (n, d)  -- null component
    x_perp = x_U_coords @ V_inv.T       # (n, k)  -- Step 6

    # ------------------------------------------------------------------ #
    # Step 8 reconstruction with x_perp unchanged (alpha=0 equivalent)
    # ------------------------------------------------------------------ #
    x_reconstructed = x_null + x_perp @ V.T @ U_prime.T  # (n, d)

    # ------------------------------------------------------------------ #
    # Reference: manual Gram-Schmidt for the two vectors
    # ------------------------------------------------------------------ #
    v1 = vectors[0] / vectors[0].norm()
    v2 = vectors[1] / vectors[1].norm()
    v2_prime = v2 - (v1 @ v2) * v1
    v2_prime = v2_prime / v2_prime.norm()

    # a2 = (<v1, a>, <v2', a>)
    a2 = torch.stack([x_batch @ v1, x_batch @ v2_prime], dim=-1)  # (n, 2)

    # a_d^a = v1<v1,a> + v2'<v2',a>
    a_d_a = (x_batch @ v1).unsqueeze(-1) * v1 + \
            (x_batch @ v2_prime).unsqueeze(-1) * v2_prime          # (n, d)

    # ------------------------------------------------------------------ #
    # Check 1: orthogonal coords preserved  a2*(0) == a2
    # x_perp should equal a2 up to the basis rotation baked into V
    # so we check via the span component instead
    # ------------------------------------------------------------------ #
    check1_ok = torch.allclose(x_U, a_d_a, atol=atol)
    if verbose:
        max_err = (x_U - a_d_a).abs().max().item()
        status = "✅ PASS" if check1_ok else "❌ FAIL"
        print(f"Check 1 — span component == Gram-Schmidt projection  {status}  (max |err| = {max_err:.2e})")

    # ------------------------------------------------------------------ #
    # Check 2: null component preserved  a_d^s*(0) == a_d^s
    # ------------------------------------------------------------------ #
    a_d_s = x_batch - a_d_a
    check2_ok = torch.allclose(x_null, a_d_s, atol=atol)
    if verbose:
        max_err = (x_null - a_d_s).abs().max().item()
        status = "✅ PASS" if check2_ok else "❌ FAIL"
        print(f"Check 2 — null component == a_d^s                   {status}  (max |err| = {max_err:.2e})")

    # ------------------------------------------------------------------ #
    # Check 3: full round-trip  a*(0) == a
    # ------------------------------------------------------------------ #
    check3_ok = torch.allclose(x_reconstructed, x_batch, atol=atol)
    if verbose:
        max_err = (x_reconstructed - x_batch).abs().max().item()
        status = "✅ PASS" if check3_ok else "❌ FAIL"
        print(f"Check 3 — full reconstruction == original            {status}  (max |err| = {max_err:.2e})")

    if verbose:
        all_pass = check1_ok and check2_ok and check3_ok
        print(f"\n{'All checks passed ✅' if all_pass else 'Some checks FAILED ❌'}")

    return {
        "check1_span_matches_gramschmidt": check1_ok,
        "check2_null_preserved":          check2_ok,
        "check3_roundtrip":               check3_ok,
    }