"""
=================================================================================
BUSINESS OPTIMIZATION USING LINEAR PROGRAMMING
Supply Chain Distribution Network Optimization
=================================================================================


Description:
This script solves a supply chain optimization problem using Linear Programming.
It minimizes transportation costs while meeting all warehouse demands and
respecting factory capacity constraints.

Requirements:
    pip install pulp numpy pandas matplotlib seaborn openpyxl

Usage:
    python business_optimization.py
=================================================================================
"""

import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ======================= CONFIGURATION =======================

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ======================= PROBLEM SETUP =======================

class SupplyChainOptimizer:
    """
    Supply Chain Distribution Network Optimizer
    
    This class implements a linear programming model to optimize
    product distribution from factories to warehouses.
    """
    
    def __init__(self):
        """Initialize the optimizer with problem parameters"""
        
        # Define factories and warehouses
        self.factories = ['Factory_A', 'Factory_B', 'Factory_C']
        self.warehouses = ['Warehouse_1', 'Warehouse_2', 'Warehouse_3', 'Warehouse_4']
        
        # Factory production capacities (units per month)
        self.capacity = {
            'Factory_A': 5000,
            'Factory_B': 6000,
            'Factory_C': 4500
        }
        
        # Warehouse demand requirements (units per month)
        self.demand = {
            'Warehouse_1': 3000,
            'Warehouse_2': 4000,
            'Warehouse_3': 3500,
            'Warehouse_4': 2500
        }
        
        # Transportation cost matrix ($ per unit)
        # Rows = Factories, Columns = Warehouses
        cost_matrix = np.array([
            [8, 6, 10, 9],   # Factory A to Warehouses 1-4
            [9, 12, 13, 7],  # Factory B to Warehouses 1-4
            [14, 9, 16, 5]   # Factory C to Warehouses 1-4
        ])
        
        self.cost_df = pd.DataFrame(
            cost_matrix, 
            index=self.factories, 
            columns=self.warehouses
        )
        
        # Initialize solution variables
        self.model = None
        self.x = None
        self.solution_df = None
        self.cost_breakdown_df = None
        self.optimal_cost = None
        
    def display_problem_info(self):
        """Display problem parameters"""
        
        print("\n" + "="*70)
        print("SUPPLY CHAIN OPTIMIZATION PROBLEM")
        print("="*70)
        
        print("\n1. FACTORY CAPACITIES:")
        for factory, cap in self.capacity.items():
            print(f"   {factory}: {cap:,} units/month")
        
        print("\n2. WAREHOUSE DEMANDS:")
        for warehouse, dem in self.demand.items():
            print(f"   {warehouse}: {dem:,} units/month")
        
        print("\n3. TRANSPORTATION COST MATRIX ($/unit):")
        print(self.cost_df)
        
        total_capacity = sum(self.capacity.values())
        total_demand = sum(self.demand.values())
        
        print(f"\n4. PROBLEM BALANCE:")
        print(f"   Total Capacity: {total_capacity:,} units")
        print(f"   Total Demand: {total_demand:,} units")
        print(f"   Status: {'BALANCED' if total_capacity == total_demand else 'EXCESS CAPACITY'}")
        print(f"   Excess: {total_capacity - total_demand:,} units")
        
    def build_model(self):
        """Build the linear programming optimization model"""
        
        print("\n" + "="*70)
        print("BUILDING OPTIMIZATION MODEL")
        print("="*70)
        
        # Create the LP problem
        self.model = pulp.LpProblem("Supply_Chain_Optimization", pulp.LpMinimize)
        
        # Decision variables: x[i,j] = units shipped from factory i to warehouse j
        self.x = pulp.LpVariable.dicts(
            "shipment",
            ((i, j) for i in self.factories for j in self.warehouses),
            lowBound=0,
            cat='Continuous'
        )
        
        # Objective function: Minimize total transportation cost
        self.model += (
            pulp.lpSum(
                self.cost_df.loc[i, j] * self.x[(i, j)] 
                for i in self.factories 
                for j in self.warehouses
            ),
            "Total_Transportation_Cost"
        )
        
        # Constraint 1: Factory capacity constraints
        for i in self.factories:
            self.model += (
                pulp.lpSum(self.x[(i, j)] for j in self.warehouses) <= self.capacity[i],
                f"Capacity_{i}"
            )
        
        # Constraint 2: Warehouse demand constraints
        for j in self.warehouses:
            self.model += (
                pulp.lpSum(self.x[(i, j)] for i in self.factories) >= self.demand[j],
                f"Demand_{j}"
            )
        
        print(f"\nModel Statistics:")
        print(f"  Decision Variables: {len(self.x)}")
        print(f"  Constraints: {len(self.model.constraints)}")
        print(f"  Objective: Minimize total transportation cost")
        
    def solve(self):
        """Solve the optimization model"""
        
        print("\n" + "="*70)
        print("SOLVING OPTIMIZATION MODEL")
        print("="*70)
        
        # Solve the model
        self.model.solve(pulp.PULP_CBC_CMD(msg=1))
        
        # Check solution status
        status = pulp.LpStatus[self.model.status]
        print(f"\nSolution Status: {status}")
        
        if self.model.status == pulp.LpStatusOptimal:
            print("✓ Optimal solution found!")
            self.extract_solution()
        else:
            print("✗ No optimal solution found!")
            
    def extract_solution(self):
        """Extract and format the optimal solution"""
        
        # Create solution matrix
        solution_matrix = np.zeros((len(self.factories), len(self.warehouses)))
        
        for i, factory in enumerate(self.factories):
            for j, warehouse in enumerate(self.warehouses):
                solution_matrix[i, j] = self.x[(factory, warehouse)].varValue
        
        # Solution DataFrame
        self.solution_df = pd.DataFrame(
            solution_matrix, 
            index=self.factories, 
            columns=self.warehouses
        )
        self.solution_df['Total_Shipped'] = self.solution_df.sum(axis=1)
        
        # Cost breakdown matrix
        cost_matrix = np.zeros((len(self.factories), len(self.warehouses)))
        for i, factory in enumerate(self.factories):
            for j, warehouse in enumerate(self.warehouses):
                cost_matrix[i, j] = solution_matrix[i, j] * self.cost_df.loc[factory, warehouse]
        
        self.cost_breakdown_df = pd.DataFrame(
            cost_matrix, 
            index=self.factories, 
            columns=self.warehouses
        )
        self.cost_breakdown_df['Total_Cost'] = self.cost_breakdown_df.sum(axis=1)
        
        # Optimal cost
        self.optimal_cost = pulp.value(self.model.objective)
        
    def display_results(self):
        """Display the optimization results"""
        
        print("\n" + "="*70)
        print("OPTIMAL SOLUTION")
        print("="*70)
        
        print("\n1. OPTIMAL SHIPMENT PLAN (units):")
        print(self.solution_df.round(2))
        
        print("\n2. COST BREAKDOWN ($):")
        print(self.cost_breakdown_df.round(2))
        
        print(f"\n{'='*70}")
        print(f"MINIMUM TOTAL TRANSPORTATION COST: ${self.optimal_cost:,.2f}")
        print(f"{'='*70}")
        
        # Calculate metrics
        total_shipped = self.solution_df.iloc[:, :-1].sum().sum()
        avg_cost_per_unit = self.optimal_cost / total_shipped
        
        print("\n3. KEY METRICS:")
        print(f"   Total Units Shipped: {total_shipped:,.0f}")
        print(f"   Average Cost per Unit: ${avg_cost_per_unit:.2f}")
        
        print("\n4. FACTORY UTILIZATION:")
        for factory in self.factories:
            shipped = self.solution_df.loc[factory, 'Total_Shipped']
            util_pct = (shipped / self.capacity[factory]) * 100
            print(f"   {factory}: {shipped:,.0f}/{self.capacity[factory]:,} units ({util_pct:.1f}%)")
        
        print("\n5. WAREHOUSE FULFILLMENT:")
        for warehouse in self.warehouses:
            received = self.solution_df[warehouse].sum()
            required = self.demand[warehouse]
            fulfillment_pct = (received / required) * 100
            print(f"   {warehouse}: {received:,.0f}/{required:,} units ({fulfillment_pct:.1f}%)")
        
    def analyze_insights(self):
        """Generate business insights"""
        
        print("\n" + "="*70)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*70)
        
        # Calculate utilization
        utilization_rates = {}
        for factory in self.factories:
            shipped = self.solution_df.loc[factory, 'Total_Shipped']
            util = (shipped / self.capacity[factory]) * 100
            utilization_rates[factory] = util
        
        most_utilized = max(utilization_rates, key=utilization_rates.get)
        least_utilized = min(utilization_rates, key=utilization_rates.get)
        
        total_capacity = sum(self.capacity.values())
        total_shipped = self.solution_df.iloc[:, :-1].sum().sum()
        excess_capacity = total_capacity - total_shipped
        
        # Find most expensive active route
        max_cost = 0
        max_route = None
        for i in self.factories:
            for j in self.warehouses:
                if self.x[(i, j)].varValue > 0:
                    cost = self.cost_df.loc[i, j]
                    if cost > max_cost:
                        max_cost = cost
                        max_route = (i, j)
        
        active_routes = sum(1 for i in self.factories for j in self.warehouses 
                          if self.x[(i, j)].varValue > 0)
        
        print("\n📊 KEY FINDINGS:")
        print(f"\n1. COST OPTIMIZATION")
        print(f"   ✓ Achieved minimum cost: ${self.optimal_cost:,.2f}")
        print(f"   ✓ Average cost: ${self.optimal_cost/total_shipped:.2f}/unit")
        
        print(f"\n2. CAPACITY ANALYSIS")
        print(f"   • Most utilized: {most_utilized} ({utilization_rates[most_utilized]:.1f}%)")
        print(f"   • Least utilized: {least_utilized} ({utilization_rates[least_utilized]:.1f}%)")
        print(f"   • Excess capacity: {excess_capacity:,.0f} units")
        
        print(f"\n3. ROUTE EFFICIENCY")
        print(f"   • Active routes: {active_routes}/{len(self.factories)*len(self.warehouses)}")
        print(f"   • Most expensive route: {max_route[0]} → {max_route[1]} (${max_cost}/unit)")
        
        print(f"\n4. DEMAND FULFILLMENT")
        print(f"   ✓ All demands met: 100%")
        
        print("\n💡 STRATEGIC RECOMMENDATIONS:")
        print(f"\n   1. Implement this optimized distribution plan to save costs")
        print(f"   2. Consider renegotiating rates for high-cost routes")
        print(f"   3. {least_utilized} has low utilization - reassess capacity needs")
        print(f"   4. {excess_capacity:,.0f} units excess provides buffer for demand spikes")
        print(f"   5. Use this model weekly/monthly for adaptive planning")
        
    def create_visualizations(self):
        """Generate comprehensive visualizations"""
        
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Main dashboard
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Shipment Heatmap
        ax1 = fig.add_subplot(gs[0, :])
        sns.heatmap(self.solution_df.iloc[:, :-1], annot=True, fmt='.0f', 
                   cmap='Blues', cbar_kws={'label': 'Units'}, ax=ax1, linewidths=0.5)
        ax1.set_title('Optimal Shipment Plan (Units)', fontsize=14, fontweight='bold')
        
        # 2. Factory Utilization
        ax2 = fig.add_subplot(gs[1, 0])
        shipped = self.solution_df['Total_Shipped'].values
        capacities = [self.capacity[f] for f in self.factories]
        x_pos = np.arange(len(self.factories))
        
        ax2.bar(x_pos - 0.2, shipped, 0.4, label='Shipped', color='#2ecc71')
        ax2.bar(x_pos + 0.2, capacities, 0.4, label='Capacity', color='#3498db', alpha=0.6)
        ax2.set_xlabel('Factory')
        ax2.set_ylabel('Units')
        ax2.set_title('Factory Capacity Utilization', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(self.factories, rotation=45)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Utilization Percentage
        ax3 = fig.add_subplot(gs[1, 1])
        util_pcts = [(s/self.capacity[f])*100 for s, f in zip(shipped, self.factories)]
        bars = ax3.bar(self.factories, util_pcts, color=['#3498db', '#9b59b6', '#e67e22'])
        ax3.set_title('Utilization Rate (%)', fontweight='bold')
        ax3.set_ylabel('Percentage (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100%')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 4. Cost Heatmap
        ax4 = fig.add_subplot(gs[1, 2])
        sns.heatmap(self.cost_breakdown_df.iloc[:, :-1], annot=True, fmt='.0f',
                   cmap='YlOrRd', cbar_kws={'label': 'Cost ($)'}, ax=ax4)
        ax4.set_title('Cost Breakdown ($)', fontweight='bold')
        
        # 5. Cost Distribution Pie
        ax5 = fig.add_subplot(gs[2, 0])
        factory_costs = self.cost_breakdown_df['Total_Cost']
        colors = ['#3498db', '#9b59b6', '#e67e22']
        ax5.pie(factory_costs, labels=self.factories, autopct='%1.1f%%', 
               startangle=90, colors=colors)
        ax5.set_title('Cost Distribution by Factory', fontweight='bold')
        
        # 6. Warehouse Demand vs Received
        ax6 = fig.add_subplot(gs[2, 1])
        received = [self.solution_df[w].sum() for w in self.warehouses]
        demands = [self.demand[w] for w in self.warehouses]
        
        x_pos = np.arange(len(self.warehouses))
        ax6.bar(x_pos - 0.2, demands, 0.4, label='Required', color='#e74c3c')
        ax6.bar(x_pos + 0.2, received, 0.4, label='Received', color='#2ecc71')
        ax6.set_xlabel('Warehouse')
        ax6.set_ylabel('Units')
        ax6.set_title('Demand vs Fulfillment', fontweight='bold')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(self.warehouses, rotation=45)
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        
        # 7. Summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        total_shipped = self.solution_df.iloc[:, :-1].sum().sum()
        summary_text = f"""
OPTIMIZATION SUMMARY
{'='*30}

Optimal Total Cost:
${self.optimal_cost:,.2f}

Total Units Shipped:
{total_shipped:,.0f} units

Average Cost per Unit:
${self.optimal_cost/total_shipped:.2f}

Status: ✓ OPTIMAL
All Demands Met: ✓
        """
        
        ax7.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
                verticalalignment='center')
        
        plt.suptitle('Supply Chain Optimization - Complete Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save
        output_path = results_dir / 'optimization_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        
        plt.show()
        
    def export_results(self):
        """Export results to Excel"""
        
        print("\n" + "="*70)
        print("EXPORTING RESULTS")
        print("="*70)
        
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        excel_path = results_dir / 'optimization_results.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Optimal Shipments
            self.solution_df.to_excel(writer, sheet_name='Optimal_Shipments')
            
            # Sheet 2: Cost Breakdown
            self.cost_breakdown_df.to_excel(writer, sheet_name='Cost_Breakdown')
            
            # Sheet 3: Summary
            total_shipped = self.solution_df.iloc[:, :-1].sum().sum()
            summary_data = {
                'Metric': [
                    'Optimal Total Cost',
                    'Total Units Shipped',
                    'Average Cost per Unit',
                    'Total Capacity',
                    'Total Demand',
                    'Excess Capacity',
                    'Capacity Utilization %'
                ],
                'Value': [
                    f'${self.optimal_cost:,.2f}',
                    f'{total_shipped:,.0f}',
                    f'${self.optimal_cost/total_shipped:.2f}',
                    f'{sum(self.capacity.values()):,}',
                    f'{sum(self.demand.values()):,}',
                    f'{sum(self.capacity.values()) - total_shipped:,.0f}',
                    f'{(total_shipped/sum(self.capacity.values())*100):.1f}%'
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 4: Factory Utilization
            util_data = []
            for factory in self.factories:
                shipped = self.solution_df.loc[factory, 'Total_Shipped']
                util_data.append({
                    'Factory': factory,
                    'Capacity': self.capacity[factory],
                    'Shipped': shipped,
                    'Unused': self.capacity[factory] - shipped,
                    'Utilization %': (shipped/self.capacity[factory])*100
                })
            util_df = pd.DataFrame(util_data)
            util_df.to_excel(writer, sheet_name='Factory_Utilization', index=False)
        
        print(f"✓ Saved: {excel_path}")
        
    def run_complete_analysis(self):
        """Run the complete optimization workflow"""
        
        print("\n" + "="*70)
        print("BUSINESS OPTIMIZATION USING LINEAR PROGRAMMING")
        print("="*70)
        
        # Display problem
        self.display_problem_info()
        
        # Build model
        self.build_model()
        
        # Solve
        self.solve()
        
        if self.model.status == pulp.LpStatusOptimal:
            # Display results
            self.display_results()
            
            # Insights
            self.analyze_insights()
            
            # Visualizations
            self.create_visualizations()
            
            # Export
            self.export_results()
            
            print("\n" + "="*70)
            print("✓ OPTIMIZATION COMPLETE!")
            print("="*70)
            print("\nCheck 'results/' folder for:")
            print("  • optimization_results.xlsx")
            print("  • optimization_dashboard.png")
        

# ======================= MAIN EXECUTION =======================

def main():
    """Main function to run the optimization"""
    
    # Create optimizer instance
    optimizer = SupplyChainOptimizer()
    
    # Run complete analysis
    optimizer.run_complete_analysis()


if __name__ == "__main__":
    main()


# ======================= END OF SCRIPT =======================
"""
Additional Notes:
-----------------
1. This script demonstrates Linear Programming for business optimization
2. The problem is a classic Transportation Problem in Operations Research
3. Can be extended to include:
   - Multiple time periods
   - Inventory costs
   - Production costs
   - Multiple products
   - Uncertainty analysis

For questions or improvements, please contribute to the GitHub repository!
"""
