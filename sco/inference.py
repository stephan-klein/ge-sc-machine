import sys
import os
import json
import argparse
from time import time

import networkx as nx
import torch
from torch import nn
from logbook import Logger, StreamHandler

from .common.utils import check_gpu, get_node_ids, get_binary_mask, \
                            get_line_numbers, get_edges, \
                            get_color_node, get_node_type, get_bug_lines
from .common.process_graphs.call_graph_generator import generate_cg
from .common.process_graphs.control_flow_graph_generator import generate_cfg
from .common.process_graphs.combination_call_graph_and_control_flow_graph_helper import combine_cfg_cg
from .config import NODE_MODEL_OPTS
from .config import GRAPH_MODEL_OPTS

# Other initialization code
torch.manual_seed(1)
# Set up the StreamHandler to output logs to stdout
StreamHandler(sys.stdout).push_application()
logger = Logger(__name__)
CATEGORIES_OF_HEATMAP = 15

# is_gpu = check_gpu()

def main(argv):
    parser = argparse.ArgumentParser(description='Smart contract vulnerability detection')
    parser.add_argument('solidity_file', type=str, help='Path to the Solidity file to analyze')
    parser.add_argument('-b', '--bugtypes', type=str, help='Comma separated list of bug types to check')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show verbose output with full details')
    

    
    # Handle both direct sys.argv passing and argparse
    if len(argv) > 1 and argv[1].startswith('-'):
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args([argv[1]] + argv[2:] if len(argv) > 1 else [])
    
    sm_path = args.solidity_file
    verbose = args.verbose
    
    # Filter bug types if specified
    bug_types = None
    if args.bugtypes:
        bug_types = args.bugtypes.split(',')
        logger.info(f"Analyzing only for bug types: {bug_types}")
    
    # Read the Solidity file content
    with open(sm_path, 'r') as f:
        sm_content = f.read()
    sm_length = len(sm_content.split('\n'))
    sm_name = os.path.basename(sm_path)

    # Generate graphs from provided smart contract file    
    cfg_graph = generate_cfg(sm_path)
    if cfg_graph is None:
        print(json.dumps({'messages': 'Found an illegal solidity smart contract'}))
        sys.exit(1)
    cg_graph = generate_cg(sm_path)
    if cg_graph is None:
        print(json.dumps({'messages': 'Found an illegal solidity smart contract'}))
        sys.exit(1)
    cfg_cg_graph = combine_cfg_cg(cfg_graph, cg_graph)

    total_reports = []
    for bug in NODE_MODEL_OPTS.keys():
        # Skip if bug types specified and this one is not in the list
        if bug_types and bug not in bug_types:
            continue
            
        node_model = NODE_MODEL_OPTS[bug]
        graph_model = GRAPH_MODEL_OPTS[bug]
        report = {'type': bug}

        # Inference Graph level
        original_graph = graph_model.nx_graph
        extra_graph = nx.disjoint_union(original_graph, cfg_cg_graph)
        begin_time = time()
        with torch.no_grad():
            try:
                logits = graph_model.extend_forward(extra_graph, [sm_name])
            except Exception as e:
                logger.info(e)
                raise e
            graph_preds = nn.functional.softmax(logits, dim=1)
            _, indices = torch.max(graph_preds, dim=1)
            graph_preds = indices.long().cpu().tolist()
            report['graph_runtime'] = int((time() - begin_time) * 1000)

        # Inference Node level
        original_graph = node_model.nx_graph
        extra_graph = nx.disjoint_union(original_graph, cfg_cg_graph)
        file_ids = get_node_ids(extra_graph, [sm_name])
        line_numbers = get_line_numbers(extra_graph, [sm_name])
        file_edges = get_edges(extra_graph, [sm_name], file_ids)
        node_type = get_node_type(extra_graph, [sm_name])
        begin_time = time()
        with torch.no_grad():
            try:
                logits, _ = node_model.extend_forward(extra_graph)
            except Exception as e:
                logger.info(e)
                raise e
            file_mask = get_binary_mask(len(extra_graph), file_ids)
            node_preds = logits[file_mask]
            node_preds = nn.functional.softmax(node_preds, dim=1)
            _, indices = torch.max(node_preds, dim=1)
            node_preds = indices.long().cpu().tolist()
        report['node_runtime'] = int((time() - begin_time) * 1000)

        if graph_preds[0] == 0:
            node_preds = [0] * len(node_preds)
        logger.info(f"Number of buggy nodes: { node_preds.count(1)}")
        report['number_of_bug_node'] = node_preds.count(1)
        report['vulnerability'] = 0 if report['number_of_bug_node'] == 0 else 1
        report['vulnerability'] = graph_preds[0]
        report['number_of_normal_node'] = node_preds.count(0)
        logger.info(f"Number of clean nodes: { node_preds.count(0)}")
        assert len(node_preds) == len(line_numbers)

        # Skip generating detailed data if not needed and not vulnerable
        if not verbose and report['vulnerability'] == 0:
            # Only include required fields for non-verbose mode
            simplified_report = {
                'type': bug,
                'vulnerability': report['vulnerability']
            }
            total_reports.append(simplified_report)
            continue

        # Only generate detailed results if verbose or the contract is vulnerable
        if verbose or report['vulnerability'] == 1:
            results = []
            for i in range(len(line_numbers)):
                _pred = node_preds[i]
                # TODO: This filter is taken over from the original code - the filter messes with the statistics but seems reasonable so we keep it for now 
                if node_type[i] in ['FUNCTION_NAME', 'CONTRACT_FUNCTION', 'ENTRY_POINT', 'END_IF']:
                    _pred = 0
                # Only include vulnerable nodes if not in verbose mode
                if verbose or _pred == 1:
                    results.append({'id': i, 'code_lines': line_numbers[i], 'vulnerability': _pred})
            report['results'] = results
        
        # Only generate graph data if verbose
        if verbose:
            color_nodes = get_color_node(extra_graph, [sm_name])
            links = [{"source": str(file_edges[i][0]), "target": str(file_edges[i][1])} for i in range(len(file_edges))]
            nodes = []
            for i in range(len(line_numbers)):
                string = ''
                for item in line_numbers[i]:
                    string += f"{item} "
                _pred = node_preds[i]
                if node_type[i] in ['FUNCTION_NAME', 'CONTRACT_FUNCTION', 'ENTRY_POINT', 'END_IF']:
                    _pred = 0
                string = f"id:{i}, node type: {node_type[i]},\ncode lines: {string}."
                node = {'id': str(i), 'name': string, 'error': _pred, 'color': color_nodes[i], 'code_lines': line_numbers[i]}
                nodes.append(node)
            graph = {"nodes": nodes, "links": links}
            report['graph'] = graph

            # Generate bug density data only in verbose mode
            bug_lines = get_bug_lines(node_preds, line_numbers)
            max_line = sm_length if len(bug_lines) == 0 else max(bug_lines)
            bug_population = torch.zeros(max_line + 1)
            bug_population[bug_lines] = 1
            bug_population = bug_population[1:sm_length]
            line_per_category = sm_length / CATEGORIES_OF_HEATMAP
            bug_density = []
            for i in range(CATEGORIES_OF_HEATMAP):
                heatmap_point = {
                    "x": f'{int(i * line_per_category) + 1}-{int((i+1) * line_per_category)}',
                    "y": bug_population[int(i * line_per_category): int((i+1) * line_per_category)].tolist().count(1)
                }
                bug_density.append(heatmap_point)
            report['bug_density'] = bug_density

        total_reports.append(report)

    # Format response based on verbose flag
    if verbose:
        response = {
            'summaries': total_reports,
            'smart_contract_length': sm_length,
            'heatmap_categories': CATEGORIES_OF_HEATMAP,
            'messages': 'OK'
        }
    else:
        # Simplified response for non-verbose mode
        response = {
            'summaries': total_reports,
            'messages': 'OK'
        }
    
    print(f"Result: {json.dumps(response)}")

if __name__ == "__main__":
    try:
        main(sys.argv)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)