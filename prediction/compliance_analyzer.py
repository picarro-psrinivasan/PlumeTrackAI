import pandas as pd
import numpy as np
import json
import sys
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prediction.concentration_predictor import load_trained_concentration_model, get_recent_concentration_data, prepare_concentration_input_sequence, predict_concentration_6hours_ahead

class ComplianceAnalyzer:
    """
    Analyzes predicted concentration data for compliance violations and impact assessment.
    """
    
    def __init__(self):
        """
        Initialize the compliance analyzer with regulatory thresholds.
        """
        # Regulatory thresholds (in PPB) - these should be updated based on actual regulations
        self.thresholds = {
            'C2H4Cl2': {
                'threshold_value': 0.5,  # PPB
                'threshold_unit': 'PPB',
                'compound_name': '1,2-Dichloroethane',
                'risk_level': 'high',
                'exposure_time_limit': 8  # hours
            },
            'C2H4O': {
                'threshold_value': 0.11,  # PPB
                'threshold_unit': 'PPB', 
                'compound_name': 'Acetaldehyde',
                'risk_level': 'medium',
                'exposure_time_limit': 8  # hours
            },
            'C2H3Cl': {
                'threshold_value': 1.174,  # PPB
                'threshold_unit': 'PPB',
                'compound_name': 'Vinyl Chloride',
                'risk_level': 'critical',
                'exposure_time_limit': 1  # hours
            },
            'C4H6': {
                'threshold_value': 1.36,  # PPB
                'threshold_unit': 'PPB',
                'compound_name': '1,3-Butadiene',
                'risk_level': 'critical',
                'exposure_time_limit': 1  # hours
            },
            'C4H5Cl': {
                'threshold_value': 0.22,  # PPB
                'threshold_unit': 'PPB',
                'compound_name': 'Chloroprene',
                'risk_level': 'high',
                'exposure_time_limit': 4  # hours
            },
            'C6H6': {
                'threshold_value': 1.0,  # PPB
                'threshold_unit': 'PPB',
                'compound_name': 'Benzene',
                'risk_level': 'critical',
                'exposure_time_limit': 1  # hours
            }
        }
        
        # Compound mapping from model output to actual names
        self.compound_mapping = {
            'compound_1': 'C2H4Cl2',
            'compound_2': 'C2H4O',
            'compound_3': 'C2H3Cl', 
            'compound_4': 'C4H6',
            'compound_5': 'C4H5Cl',
            'compound_6': 'C6H6'
        }
    
    def analyze_compliance_impact(self, predictions: List[Dict]) -> Dict:
        """
        Analyze compliance impact of predicted concentrations.
        
        Args:
            predictions: List of prediction dictionaries from concentration model
            
        Returns:
            Dict containing compliance analysis results
        """
        print("=== Compliance Impact Analysis ===")
        
        compliance_results = {
            'summary': {},
            'time_step_analysis': [],
            'violations': [],
            'risk_assessment': {},
            'recommendations': []
        }
        
        # Analyze each time step (15-minute intervals)
        for step_pred in predictions:
            step = step_pred['step']
            time_minutes = step_pred['time_minutes']
            compounds = step_pred['compounds']
            
            # Convert minutes to readable time
            hours = time_minutes // 60
            mins = time_minutes % 60
            time_str = f"{hours:02d}:{mins:02d}"
            
            step_analysis = {
                'step': step,
                'time_minutes': time_minutes,
                'time_str': time_str,
                'compounds': {},
                'violations': [],
                'risk_score': 0
            }
            
            # Analyze each compound
            for compound_key, concentration in compounds.items():
                compound_name = self.compound_mapping.get(compound_key, compound_key)
                threshold_info = self.thresholds.get(compound_name, {})
                
                threshold_value = threshold_info.get('threshold_value', float('inf'))
                risk_level = threshold_info.get('risk_level', 'unknown')
                compound_display_name = threshold_info.get('compound_name', compound_name)
                
                # Calculate violation percentage
                violation_percentage = (concentration / threshold_value) * 100 if threshold_value > 0 else 0
                is_violation = concentration > threshold_value
                
                compound_analysis = {
                    'compound_name': compound_display_name,
                    'predicted_concentration': concentration,
                    'threshold_value': threshold_value,
                    'violation_percentage': violation_percentage,
                    'is_violation': is_violation,
                    'risk_level': risk_level,
                    'unit': 'PPB'
                }
                
                step_analysis['compounds'][compound_name] = compound_analysis
                
                # Track violations
                if is_violation:
                    violation_info = {
                        'step': step,
                        'time_minutes': time_minutes,
                        'time_str': time_str,
                        'compound': compound_display_name,
                        'concentration': concentration,
                        'threshold': threshold_value,
                        'excess': concentration - threshold_value,
                        'risk_level': risk_level
                    }
                    step_analysis['violations'].append(violation_info)
                    compliance_results['violations'].append(violation_info)
                    
                    # Calculate risk score (higher for critical compounds)
                    risk_multiplier = {'low': 1, 'medium': 2, 'high': 3, 'critical': 5}
                    step_analysis['risk_score'] += (concentration / threshold_value) * risk_multiplier.get(risk_level, 1)
            
            compliance_results['time_step_analysis'].append(step_analysis)
        
        # Generate summary statistics
        compliance_results['summary'] = self._generate_summary(compliance_results)
        
        # Generate risk assessment
        compliance_results['risk_assessment'] = self._assess_overall_risk(compliance_results)
        
        # Generate recommendations
        compliance_results['recommendations'] = self._generate_recommendations(compliance_results)
        
        return compliance_results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """
        Generate summary statistics for compliance analysis.
        """
        total_violations = len(results['violations'])
        compounds_with_violations = set()
        critical_violations = 0
        
        for violation in results['violations']:
            compounds_with_violations.add(violation['compound'])
            if violation['risk_level'] == 'critical':
                critical_violations += 1
        
        return {
            'total_violations': total_violations,
            'compounds_with_violations': len(compounds_with_violations),
            'critical_violations': critical_violations,
            'prediction_time_steps': len(results['time_step_analysis']),
            'prediction_hours': len(results['time_step_analysis']) // 4,  # 4 time steps per hour
            'compliance_status': 'NON-COMPLIANT' if total_violations > 0 else 'COMPLIANT'
        }
    
    def _assess_overall_risk(self, results: Dict) -> Dict:
        """
        Assess overall risk based on violations and concentrations.
        """
        if not results['violations']:
            return {
                'overall_risk': 'LOW',
                'risk_score': 0,
                'primary_concerns': [],
                'critical_compounds': [],
                'immediate_actions_needed': False
            }
        
        # Calculate risk metrics
        max_violation_excess = max([v['excess'] for v in results['violations']])
        critical_compounds = [v['compound'] for v in results['violations'] if v['risk_level'] == 'critical']
        avg_risk_score = np.mean([s['risk_score'] for s in results['time_step_analysis'] if s['risk_score'] > 0])
        
        # Determine overall risk level
        if critical_compounds or avg_risk_score > 3:
            overall_risk = 'CRITICAL'
        elif avg_risk_score > 2:
            overall_risk = 'HIGH'
        elif avg_risk_score > 1:
            overall_risk = 'MEDIUM'
        else:
            overall_risk = 'LOW'
        
        return {
            'overall_risk': overall_risk,
            'risk_score': avg_risk_score,
            'max_violation_excess': max_violation_excess,
            'critical_compounds': critical_compounds,
            'immediate_actions_needed': len(critical_compounds) > 0
        }
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """
        Generate recommendations based on compliance analysis.
        """
        recommendations = []
        
        if not results['violations']:
            recommendations.append("All predicted concentrations are within regulatory limits.")
            recommendations.append("Continue monitoring to maintain compliance.")
            return recommendations
        
        # Analyze violations and generate specific recommendations
        critical_violations = [v for v in results['violations'] if v['risk_level'] == 'critical']
        high_violations = [v for v in results['violations'] if v['risk_level'] == 'high']
        
        if critical_violations:
            recommendations.append("CRITICAL: Immediate action required for critical compound violations!")
            for violation in critical_violations:
                recommendations.append(f"   - {violation['compound']}: {violation['concentration']:.3f} PPB (threshold: {violation['threshold']:.3f} PPB)")
        
        if high_violations:
            recommendations.append("HIGH RISK: Address high-risk compound violations promptly.")
            for violation in high_violations:
                recommendations.append(f"   - {violation['compound']}: {violation['concentration']:.3f} PPB (threshold: {violation['threshold']:.3f} PPB)")
        
        # General recommendations
        recommendations.append("Investigate potential sources of elevated concentrations.")
        recommendations.append("Review emission control systems and maintenance schedules.")
        recommendations.append("Notify relevant personnel and stakeholders.")
        recommendations.append("Document all violations and mitigation actions taken.")
        
        return recommendations
    
    def print_compliance_report(self, results: Dict):
        """
        Print a formatted compliance report.
        """
        print("\n" + "="*80)
        print("                    COMPLIANCE IMPACT ANALYSIS REPORT")
        print("="*80)
        
        # Summary
        summary = results['summary']
        print(f"\nSUMMARY:")
        print(f"   Compliance Status: {summary['compliance_status']}")
        print(f"   Total Violations: {summary['total_violations']}")
        print(f"   Compounds with Violations: {summary['compounds_with_violations']}")
        print(f"   Critical Violations: {summary['critical_violations']}")
        
        # Risk Assessment
        risk = results['risk_assessment']
        print(f"\nRISK ASSESSMENT:")
        print(f"   Overall Risk: {risk['overall_risk']}")
        print(f"   Risk Score: {risk['risk_score']:.2f}")
        if risk['critical_compounds']:
            print(f"   Critical Compounds: {', '.join(risk['critical_compounds'])}")
        
        # Time Step Analysis
        print(f"\n15-MINUTE INTERVAL COMPLIANCE ANALYSIS:")
        for step_analysis in results['time_step_analysis']:
            time_str = step_analysis['time_str']
            violations = step_analysis['violations']
            
            if violations:
                print(f"\n   Time {time_str} (Step {step_analysis['step']}):")
                for violation in violations:
                    print(f"   {violation['compound']}: {violation['concentration']:.3f} PPB (threshold: {violation['threshold']:.3f} PPB)")
            else:
                print(f"   Time {time_str} (Step {step_analysis['step']}): All compounds within limits")
        
        # Recommendations
        print(f"\n RECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"   {rec}")
        
        print("\n" + "="*80)

def main():
    """
    Main function to run compliance analysis on predicted concentrations.
    """
    print("=== PlumeTrackAI Compliance Impact Analyzer ===")
    
    # Get concentration predictions using the new function
    print("\nGetting concentration predictions...")
    from prediction.concentration_predictor import get_concentration_predictions
    
    prediction_results = get_concentration_predictions()
    
    if prediction_results is None:
        print("Error: Could not get concentration predictions.")
        return
    
    predictions = prediction_results['predictions']
    
    # Analyze compliance impact
    print("\nAnalyzing compliance impact...")
    analyzer = ComplianceAnalyzer()
    compliance_results = analyzer.analyze_compliance_impact(predictions)
    
    # Print comprehensive report
    analyzer.print_compliance_report(compliance_results)
    
    print("\nCompliance analysis completed!")

if __name__ == "__main__":
    main() 