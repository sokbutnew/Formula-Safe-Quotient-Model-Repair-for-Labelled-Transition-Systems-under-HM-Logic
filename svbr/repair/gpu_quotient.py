from __future__ import annotations

from collections import OrderedDict

from svbr.repair.add_delete import Edge, QuotientSignature, RepairLTS


def strong_v_quotient_torch(
    model: RepairLTS,
    v_actions: set[str],
    device: str = "cuda",
    strict_device: bool = False,
) -> QuotientSignature:
    """Compute the same strong-V quotient with tensor-heavy steps on torch.

    The exact partition dictionary is still built on CPU because HML labels are
    strings and signatures are variable-length sets. The expensive edge
    projection/unique operations are performed on the requested torch device.
    """
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        message = f"Requested quotient device '{device}', but CUDA is not available."
        if strict_device:
            raise SystemExit(message)
        print("Warning:", message, "Using CPU for quotient tensors.")
        device = "cpu"
    torch_device = torch.device(device)

    visible_edges = [edge for edge in model.edges if edge.action not in v_actions]
    if not visible_edges:
        return QuotientSignature(tuple(0 for _ in range(model.state_count)), frozenset())

    try:
        with torch.no_grad():
            actions = sorted({edge.action for edge in visible_edges})
            action_to_id = {action: index + 1 for index, action in enumerate(actions)}
            src = torch.tensor([edge.src for edge in visible_edges], dtype=torch.long, device=torch_device)
            dst = torch.tensor([edge.dst for edge in visible_edges], dtype=torch.long, device=torch_device)
            action_ids = torch.tensor([action_to_id[edge.action] for edge in visible_edges], dtype=torch.long, device=torch_device)

            state_count = int(model.state_count)
            block = torch.zeros(state_count, dtype=torch.long, device=torch_device)
            max_edge_code = (len(actions) + 1) * (state_count + 1) + 1

            while True:
                dst_blocks = block[dst]
                edge_codes = action_ids * (state_count + 1) + dst_blocks + 1
                pair_codes = src * max_edge_code + edge_codes
                unique_pairs = torch.unique(pair_codes, sorted=True)
                unique_src = torch.div(unique_pairs, max_edge_code, rounding_mode="floor").cpu().tolist()
                unique_edge_codes = (unique_pairs % max_edge_code).cpu().tolist()

                signature_to_block: OrderedDict[tuple[int, ...], int] = OrderedDict()
                next_block_list = [0] * state_count
                pair_index = 0
                pair_count = len(unique_src)
                for state in range(state_count):
                    signature = []
                    while pair_index < pair_count and unique_src[pair_index] == state:
                        signature.append(int(unique_edge_codes[pair_index]))
                        pair_index += 1
                    key = tuple(signature)
                    if key not in signature_to_block:
                        signature_to_block[key] = len(signature_to_block)
                    next_block_list[state] = signature_to_block[key]

                next_block = torch.tensor(next_block_list, dtype=torch.long, device=torch_device)
                is_stable = torch.equal(next_block, block)
                del dst_blocks, edge_codes, pair_codes, unique_pairs, unique_src, unique_edge_codes
                if is_stable:
                    final_block = tuple(int(item) for item in next_block_list)
                    quotient_edges = {
                        (final_block[edge.src], edge.action, final_block[edge.dst])
                        for edge in visible_edges
                    }
                    result = QuotientSignature(final_block, frozenset(quotient_edges))
                    del src, dst, action_ids, block, next_block
                    return result
                old_block = block
                block = next_block
                del old_block
    finally:
        if torch_device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


__all__ = ["strong_v_quotient_torch"]
