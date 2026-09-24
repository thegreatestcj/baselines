#!/usr/bin/env python3
"""
Evaluate physics parameter prediction accuracy for specific scenario IDs.
"""

import json
import numpy as np
import os
import argparse

def load_json(path):
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)

def calculate_error(gt_value, pred_value):
    """Calculate relative error percentage"""
    if abs(gt_value) < 1e-10:
        return abs(pred_value - gt_value) * 100
    return abs(pred_value - gt_value) / abs(gt_value) * 100

def calculate_velocity_error(gt_vel, pred_vel):
    """Calculate velocity magnitude error percentage"""
    gt_mag = np.linalg.norm(gt_vel)
    pred_mag = np.linalg.norm(pred_vel)
    if gt_mag < 1e-10:
        return pred_mag * 100
    return abs(pred_mag - gt_mag) / gt_mag * 100

def evaluate_scenario(gt_path, pred_path, scenario_id):
    """Evaluate a single scenario and return errors"""
    metadata_path = os.path.join(gt_path, scenario_id, 'metadata.json')
    pred_json_path = os.path.join(pred_path, scenario_id, f'{scenario_id}-pred.json')

    if not os.path.exists(metadata_path):
        print(f"  Warning: metadata.json not found for {scenario_id}")
        print(f"    Looked in: {metadata_path}")
        # Check what files are in the directory
        scenario_dir = os.path.join(gt_path, scenario_id)
        if os.path.exists(scenario_dir):
            files = os.listdir(scenario_dir)
            print(f"    Files in {scenario_dir}: {files[:5]}...")  # Show first 5 files
        else:
            print(f"    Directory doesn't exist: {scenario_dir}")
        return None

    if not os.path.exists(pred_json_path):
        print(f"  Warning: pred.json not found for {scenario_id}")
        print(f"    Looked in: {pred_json_path}")
        return None

    # Load files
    metadata = load_json(metadata_path)
    pred_data = load_json(pred_json_path)

    errors = {}

    # Process each object
    for obj_num in [1, 2]:
        gt_obj_key = f'obj{obj_num}'

        # Get GT data
        if gt_obj_key not in metadata:
            continue

        gt_obj = metadata[gt_obj_key]

        # Get predicted data
        pred_obj = None
        if 'sub_objects' in pred_data:
            for obj in pred_data['sub_objects']:
                if obj['object_id'] == obj_num:
                    pred_obj = obj
                    break

        if pred_obj is None:
            continue

        obj_errors = {}

        # Velocity error
        if 'initial_velocity' in gt_obj and 'vel' in pred_obj:
            obj_errors['velocity'] = calculate_velocity_error(
                gt_obj['initial_velocity'],
                pred_obj['vel']
            )

        # Material parameters
        if 'mat_params' in pred_obj:
            mat = pred_obj['mat_params']

            # Common parameters
            if 'rho' in gt_obj and 'rho' in mat:
                obj_errors['rho'] = calculate_error(gt_obj['rho'], mat['rho'])

            # Material-specific parameters
            material_type = gt_obj.get('material', '')

            if material_type in ['elastic', 'plastic']:
                if 'E' in gt_obj and 'E' in mat:
                    obj_errors['E'] = calculate_error(gt_obj['E'], mat['E'])
                if 'nu' in gt_obj and 'nu' in mat:
                    obj_errors['nu'] = calculate_error(gt_obj['nu'], mat['nu'])

                if material_type == 'plastic' and 'yield_stress' in mat:
                    if 'yield_stress' in gt_obj:
                        # Convert von Mises yield stress to compare with GT
                        # The predicted value needs to be divided by sqrt(2/3)
                        pred_yield_converted = mat['yield_stress'] / np.sqrt(2.0/3.0)
                        obj_errors['yield_stress'] = calculate_error(
                            gt_obj['yield_stress'],
                            pred_yield_converted
                        )

            elif material_type == 'fluid':
                # Check if GT has mu/kappa or needs conversion from E/nu
                if 'mu' in gt_obj and 'mu' in mat:
                    obj_errors['mu'] = calculate_error(gt_obj['mu'], mat['mu'])
                elif 'E' in gt_obj and 'nu' in gt_obj and 'mu' in mat:
                    # Convert E,nu to mu for comparison
                    gt_mu = gt_obj['E'] / (2 * (1 + gt_obj['nu']))
                    obj_errors['mu'] = calculate_error(gt_mu, mat['mu'])

                if 'kappa' in gt_obj and 'kappa' in mat:
                    obj_errors['kappa'] = calculate_error(gt_obj['kappa'], mat['kappa'])
                elif 'E' in gt_obj and 'nu' in gt_obj and 'kappa' in mat:
                    # Convert E,nu to kappa for comparison
                    gt_kappa = gt_obj['E'] / (3 * (1 - 2 * gt_obj['nu']))
                    obj_errors['kappa'] = calculate_error(gt_kappa, mat['kappa'])

            elif material_type == 'sand':
                if 'E' in gt_obj and 'E' in mat:
                    obj_errors['E'] = calculate_error(gt_obj['E'], mat['E'])
                if 'nu' in gt_obj and 'nu' in mat:
                    obj_errors['nu'] = calculate_error(gt_obj['nu'], mat['nu'])
                if 'friction_alpha' in gt_obj and 'friction_alpha' in mat:
                    obj_errors['friction_alpha'] = calculate_error(
                        gt_obj['friction_alpha'],
                        mat['friction_alpha']
                    )

        errors[gt_obj_key] = obj_errors

    return errors

def main():
    parser = argparse.ArgumentParser(description="Evaluate physics parameter predictions")
    parser.add_argument("--gt_path", type=str, required=True,
                        help="Path to ground truth directory")
    parser.add_argument("--pred_path", type=str, required=True,
                        help="Path to prediction directory")
    parser.add_argument("--ids", nargs='+', required=True,
                        help="Scenario IDs to evaluate (e.g., 01 02 03)")

    args = parser.parse_args()

    print(f"\nEvaluating scenarios: {', '.join(args.ids)}")
    print("="*60)

    # Store all errors for averaging
    all_errors = {}

    # Evaluate each scenario
    for scenario_id in args.ids:
        print(f"\nScenario {scenario_id}:")
        errors = evaluate_scenario(args.gt_path, args.pred_path, scenario_id)

        if errors is None:
            continue

        # Print errors for this scenario
        for obj_key, obj_errors in errors.items():
            if obj_errors:
                print(f"  {obj_key}:")
                for param, error in obj_errors.items():
                    print(f"    {param:15s}: {error:8.2f}%")

                    # Store for averaging
                    if param not in all_errors:
                        all_errors[param] = []
                    all_errors[param].append(error)

    # Print average errors
    if all_errors:
        print("\n" + "="*60)
        print("AVERAGE ERRORS:")
        print("="*60)

        for param in sorted(all_errors.keys()):
            errors = all_errors[param]
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            print(f"  {param:15s}: {mean_error:8.2f}% ± {std_error:8.2f}% (n={len(errors)})")

        # Overall average
        all_values = []
        for errors in all_errors.values():
            all_values.extend(errors)

        if all_values:
            print(f"\n  {'OVERALL':15s}: {np.mean(all_values):8.2f}% ± {np.std(all_values):8.2f}%")

if __name__ == "__main__":
    main()