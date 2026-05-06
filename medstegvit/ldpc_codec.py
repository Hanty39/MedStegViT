# =============================================================================
# ldpc_codec.py — LDPC kódování a dekódování payloadu (soft-decision)
#
# Implementace:
#   - PEG (Progressive Edge Growth) konstrukce parity-check matice
#   - Gaussova eliminace pro nalezení generátorové matice
#   - Min-Sum belief propagation dekodér (numericky stabilní)
#
# Konfigurace:
#   n = 512 (délka kódového slova = payload_bits)
#   k = data bits (závisí na zvoleném rate)
# =============================================================================

import numpy as np
import hashlib


class LDPCCodec:

    def __init__(self, n=512, rate=0.50, d_v=3, seed=42, max_iter=200):
        self.n = n
        self.max_iter = max_iter

        # Počet parity-check rovnic (řádků H matice)
        self.m = n - int(n * rate)
        # Váha řádku H matice (odvozená z d_v a rozměrů)
        self.d_c = int(np.ceil(d_v * n / self.m))

        # Vytvoří parity-check matici H pomocí PEG algoritmu
        self.H = self._build_peg_matrix(n, self.m, d_v, seed)

        # Z H odvodí generátorovou matici G v systematickém tvaru
        # G má tvar [I_k | P] kde I_k je jednotková matice a P je paritní část
        self.G, self.k, self._perm = self._find_systematic_generator(self.H)

        print(f"  LDPC kód: n={self.n}, k={self.k}, rate={self.k/self.n:.3f}, "
              f"d_v={d_v}, max_iter={max_iter}")

    # -----------------------------------------------------------------
    # Konstrukce parity-check matice (PEG algoritmus)
    # -----------------------------------------------------------------

    def _build_peg_matrix(self, n, m, d_v, seed):
        rng = np.random.RandomState(seed)
        H = np.zeros((m, n), dtype=np.int8)

        for j in range(n):
            for t in range(d_v):
                if t == 0:
                    # První hrana: připoj k řádku s nejnižší vahou
                    row_weights = H.sum(axis=1)
                    min_weight = row_weights.min()
                    candidates = np.where(row_weights == min_weight)[0]
                    chosen = rng.choice(candidates)
                else:
                    # Další hrany: maximalizuj girth (vzdálenost v grafu)
                    # Zjednodušená verze: vyber řádek s nejnižší vahou
                    # který ještě není připojený k tomuto sloupci
                    row_weights = H.sum(axis=1).astype(float)
                    # Řádky již připojené k j mají vysokou "cenu"
                    connected = np.where(H[:, j] > 0)[0]
                    row_weights[connected] = 1e6

                    # BFS penalizace: sousedi připojených řádků mají vyšší cenu
                    for r in connected:
                        neighbors = np.where(H[r, :] > 0)[0]
                        for nb in neighbors:
                            if nb != j:
                                shared_rows = np.where(H[:, nb] > 0)[0]
                                row_weights[shared_rows] += 0.5

                    min_w = row_weights.min()
                    candidates = np.where(np.abs(row_weights - min_w) < 0.01)[0]
                    chosen = rng.choice(candidates)

                H[chosen, j] = 1

        return H

    # -----------------------------------------------------------------
    # Generátorová matice (Gaussova eliminace)
    # -----------------------------------------------------------------

    def _find_systematic_generator(self, H):
        """
        Najde generátorovou matici G v systematickém tvaru.
        """
        m, n = H.shape
        # Pracujeme nad GF(2) — vše mod 2
        M = H.astype(np.int64).copy()
        perm = list(range(n))  # sleduje permutace sloupců

        # Gaussova eliminace s pivotováním sloupců
        pivot_row = 0
        for col_target in range(m):
            if pivot_row >= m:
                break

            # Najdi nenulový pivot v aktuálním řádku
            found = False
            for c in range(pivot_row, n):
                if M[col_target, c] != 0:
                    # Prohoď sloupce
                    M[:, [pivot_row, c]] = M[:, [c, pivot_row]]
                    perm[pivot_row], perm[c] = perm[c], perm[pivot_row]
                    found = True
                    break
            if not found:
                continue

            # Eliminuj ostatní řádky
            for r in range(m):
                if r != col_target and M[r, pivot_row] != 0:
                    M[r] = (M[r] + M[col_target]) % 2

            pivot_row += 1

        rank = pivot_row
        k = n - rank  # počet datových bitů

        # Extrahuj paritní podmatici A (z řádkově redukované H)
        # H_reduced = [I_rank | A] po permutaci
        A = M[:rank, rank:].copy()  # [rank × k]

        # Generátorová matice: G = [A^T | I_k] (v permutovaném pořadí)
        G = np.zeros((k, n), dtype=np.int8)
        G[:, :rank] = A.T % 2    # paritní část
        G[:, rank:] = np.eye(k, dtype=np.int8)  # datová část (systematická)

        # Ověření: H_perm @ G^T = 0 mod 2
        check = (M @ G.T) % 2
        if check.sum() != 0:
            print(f"  LDPC: G×H^T != 0, {check.sum()} nenulových prvků")

        return G, k, perm

    # -----------------------------------------------------------------
    # Kódování
    # -----------------------------------------------------------------

    def encode(self, data_bits):
        """
        Zakóduje k datových bitů na n-bitové kódové slovo.
        """
        data = np.asarray(data_bits, dtype=np.int8).flatten()
        if len(data) < self.k:
            # Doplní nulami pokud je kratší
            data = np.concatenate([data, np.zeros(self.k - len(data), dtype=np.int8)])
        elif len(data) > self.k:
            data = data[:self.k]

        # Kódové slovo = G^T @ data mod 2 (v permutovaném pořadí)
        codeword_perm = (self.G.T @ data) % 2

        # Depermutuj zpět do původního pořadí sloupců
        codeword = np.zeros(self.n, dtype=np.int8)
        for i, p in enumerate(self._perm):
            codeword[p] = codeword_perm[i]

        return codeword

    # -----------------------------------------------------------------
    # Soft-decision dekódování (Min-Sum Belief Propagation)
    # -----------------------------------------------------------------

    def decode_soft(self, logits, max_iter=None):
        """
        Soft-decision LDPC dekódování pomocí Min-Sum BP.
        """
        if max_iter is None:
            max_iter = self.max_iter

        logits = np.asarray(logits, dtype=np.float64).flatten()

        # Převod logitů na LLR (Log-Likelihood Ratio)
        # LLR > 0 → bit je pravděpodobně 0
        # LLR < 0 → bit je pravděpodobně 1
        # Konvence: LLR = log(P(bit=0)/P(bit=1)) = -logit
        channel_llr = -logits.copy()

        # Clamp pro numerickou stabilitu
        channel_llr = np.clip(channel_llr, -30.0, 30.0)

        H = self.H

        # Najdi nenulové pozice v H (hrany Tanner grafu)
        check_nodes = []  # pro každý check node: seznam připojených var nodes
        var_nodes = []    # pro každý var node: seznam připojených check nodes

        for j in range(self.m):
            check_nodes.append(np.where(H[j, :] != 0)[0])
        for i in range(self.n):
            var_nodes.append(np.where(H[:, i] != 0)[0])

        # Inicializace zpráv variable→check
        # v2c[j][i] = zpráva z variable i do check j
        v2c = {}
        for j in range(self.m):
            for i in check_nodes[j]:
                v2c[(j, i)] = channel_llr[i]

        # BP iterace
        converged = False
        for iteration in range(max_iter):

            # ── Check node update (Min-Sum) ───────────────────────
            # Pro každý check j a každou připojenou variable i:
            # c2v[j→i] = (znaménko) × min(|v2c[j→i']|) pro i'≠i
            c2v = {}
            for j in range(self.m):
                connected = check_nodes[j]
                if len(connected) == 0:
                    continue

                # Shromáždi zprávy od všech připojených variable nodes
                msgs = np.array([v2c[(j, i)] for i in connected])
                signs = np.sign(msgs)
                abs_msgs = np.abs(msgs)

                # Scaling factor pro min-sum (zlepšuje přesnost)
                alpha = 0.8

                for idx, i in enumerate(connected):
                    # Součin znamének KROMĚ idx
                    other_signs = np.prod(signs[:idx]) * np.prod(signs[idx+1:])
                    # Minimum absolutních hodnot KROMĚ idx
                    other_abs = np.concatenate([abs_msgs[:idx], abs_msgs[idx+1:]])
                    if len(other_abs) > 0:
                        min_abs = other_abs.min()
                    else:
                        min_abs = 0.0

                    c2v[(j, i)] = alpha * other_signs * min_abs

            # ── Variable node update ──────────────────────────────
            # Pro každou variable i a každý připojený check j:
            # v2c[i→j] = channel_llr[i] + sum(c2v[j'→i]) pro j'≠j
            for i in range(self.n):
                connected = var_nodes[i]
                if len(connected) == 0:
                    continue

                # Suma c2v zpráv od všech check nodes
                total_c2v = sum(c2v.get((j, i), 0.0) for j in connected)

                for j in connected:
                    v2c[(j, i)] = channel_llr[i] + total_c2v - c2v.get((j, i), 0.0)

            # ── Celkový LLR a hard decision ───────────────────────
            total_llr = channel_llr.copy()
            for i in range(self.n):
                for j in var_nodes[i]:
                    total_llr[i] += c2v.get((j, i), 0.0)

            hard = (total_llr < 0).astype(np.int8)

            # Syndrom check: H @ hard = 0 mod 2 → konvergence
            syndrome = (H @ hard) % 2
            if syndrome.sum() == 0:
                converged = True
                break

        # Extrahuj datové bity (depermutuj + vezmi systematickou část)
        hard_perm = np.zeros(self.n, dtype=np.int8)
        for i, p in enumerate(self._perm):
            hard_perm[i] = hard[p]

        data_bits = hard_perm[self.m:].copy()  # systematická část = data

        return data_bits[:self.k], converged

    def decode_hard(self, bits, max_iter=None):
        """
        Hard-decision dekódování (konvertuje bity na pseudo-logity).
        """
        # Převod tvrdých bitů na pseudo-logity: 0 → -1.0, 1 → +1.0
        logits = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
        return self.decode_soft(logits, max_iter)


# =====================================================================
# Funkce kompatibilní s rozhraním rs_codec.py
# =====================================================================

# Cache kodeku — jeden per (n, rate) kombinaci
_codec_cache = {}


def get_codec(n=512, rate=0.50, seed=42):
    """Vrátí LDPC kodek z cache (jeden per (n, rate) kombinaci)."""
    key = (n, round(rate, 3))
    if key not in _codec_cache:
        _codec_cache[key] = LDPCCodec(n=n, rate=rate, seed=seed)
    return _codec_cache[key]


def ldpc_encode(data: bytes, n=512, rate=0.50) -> np.ndarray:
    """
    Zakóduje datové bajty na 512-bitové LDPC kódové slovo.
    """
    codec = get_codec(n=n, rate=rate)
    # Převod bajtů na bity
    data_bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    # Ořízni na k bitů
    if len(data_bits) > codec.k:
        data_bits = data_bits[:codec.k]
    elif len(data_bits) < codec.k:
        data_bits = np.concatenate([data_bits, np.zeros(codec.k - len(data_bits), dtype=np.uint8)])

    codeword = codec.encode(data_bits)
    return codeword.astype(np.uint8)


def ldpc_decode_soft(logits, n=512, rate=0.50) -> bytes:
    """
    Soft-decision LDPC dekódování — hlavní funkce pro MedStegViT.
    """
    codec = get_codec(n=n, rate=rate)

    # Převod torch tensor na numpy pokud potřeba
    if hasattr(logits, 'detach'):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits, dtype=np.float64).flatten()

    data_bits, converged = codec.decode_soft(logits)

    if not converged:
        raise ValueError(
            f"LDPC BP nekonvergoval po {codec.max_iter} iteracích — "
            f"příliš mnoho chyb pro opravu"
        )

    # Převod bitů na bajty
    # Doplní na násobek 8
    n_bytes = (len(data_bits) + 7) // 8
    padded = np.zeros(n_bytes * 8, dtype=np.uint8)
    padded[:len(data_bits)] = data_bits
    decoded_bytes = np.packbits(padded).tobytes()

    return decoded_bytes


def ldpc_decode_hard(bits, n=512, rate=0.50) -> bytes:
    """Hard-decision dekódování (méně efektivní než soft)."""
    codec = get_codec(n=n, rate=rate)

    if hasattr(bits, 'detach'):
        bits = bits.detach().cpu().numpy()
    bits = np.asarray(bits, dtype=np.float64).flatten()

    data_bits, converged = codec.decode_hard(bits)

    if not converged:
        raise ValueError("LDPC BP nekonvergoval — příliš mnoho chyb")

    n_bytes = (len(data_bits) + 7) // 8
    padded = np.zeros(n_bytes * 8, dtype=np.uint8)
    padded[:len(data_bits)] = data_bits
    return np.packbits(padded).tobytes()
