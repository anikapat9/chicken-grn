import plotly.graph_objs as go

def plot_network(adj_matrix, gene_names):
    edges_x, edges_y = adj_matrix.nonzero()
    edge_trace = go.Scatter3d(
        x=edges_x, y=edges_y, z=[0]*len(edges_x),
        mode='lines', line=dict(width=2, color='gray')
    )
    node_trace = go.Scatter3d(
        x=range(len(gene_names)), y=range(len(gene_names)), z=[0]*len(gene_names),
        mode='markers+text', text=gene_names, marker=dict(size=5)
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.show()
