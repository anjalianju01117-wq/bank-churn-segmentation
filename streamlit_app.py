import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Segmentation",
    page_icon="🏦",
    layout="wide"
)

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE      = "#1F4E79"
LIGHT     = "#2E75B6"
ACCENT    = "#E74C3C"
GREEN     = "#27AE60"
BG        = "#F0F4F8"
GOLD      = "#F39C12"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFB; }
    [data-testid="stSidebar"] { background-color: #1F4E79; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label { color: white !important; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #1F4E79;
        margin-bottom: 10px;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; color: #1F4E79; }
    .metric-card p  { margin: 4px 0 0; color: #555; font-size: 0.9rem; }
    .red-card  { border-left-color: #E74C3C !important; }
    .red-card h2 { color: #E74C3C !important; }
    .green-card { border-left-color: #27AE60 !important; }
    .green-card h2 { color: #27AE60 !important; }
    .gold-card  { border-left-color: #F39C12 !important; }
    .gold-card h2 { color: #F39C12 !important; }
    .section-title {
        font-size: 1.3rem; font-weight: 700;
        color: #1F4E79; margin: 24px 0 12px;
        border-bottom: 2px solid #2E75B6;
        padding-bottom: 6px;
    }
    .insight-box {
        background: #EBF3FB; border-left: 4px solid #2E75B6;
        border-radius: 8px; padding: 12px 16px;
        margin: 12px 0; color: #1F4E79; font-size: 0.92rem;
    }
    .warning-box {
        background: #FDECEA; border-left: 4px solid #E74C3C;
        border-radius: 8px; padding: 12px 16px;
        margin: 12px 0; color: #922B21; font-size: 0.92rem;
    }
    .stPlotlyChart, .element-container { margin-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load & prepare data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank (2).csv")
    df["AgeGroup"]   = pd.cut(df["Age"],    bins=[0,30,45,60,100],
                               labels=["Under 30","30–45","46–60","Over 60"])
    df["CreditBand"] = pd.cut(df["CreditScore"], bins=[0,550,700,851],
                               labels=["Low (<550)","Medium (550–700)","High (>700)"])
    df["TenureGroup"]= pd.cut(df["Tenure"],  bins=[-1,2,5,10],
                               labels=["New (0–2yr)","Mid (3–5yr)","Long (6+yr)"])
    df["BalanceSeg"] = pd.cut(df["Balance"], bins=[-1,1,50000,300000],
                               labels=["Zero Balance","Low (<€50k)","High (€50k+)"])
    df["ChurnLabel"] = df["Exited"].map({1:"Churned", 0:"Retained"})
    return df

df = load_data()

# ── Sidebar navigation & filters ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Navigation")
    page = st.radio("", [
        "📊 Overview Dashboard",
        "🌍 Geographic Analysis",
        "👥 Demographic Analysis",
        "💰 Financial Segmentation",
        "🔍 Segment Explorer"
    ])
    st.markdown("---")
    st.markdown("## ⚙️ Filters")
    geo_filter  = st.multiselect("Country", ["France","Germany","Spain"],
                                  default=["France","Germany","Spain"])
    gen_filter  = st.multiselect("Gender",  ["Male","Female"],
                                  default=["Male","Female"])
    age_filter  = st.multiselect("Age Group",
                                  ["Under 30","30–45","46–60","Over 60"],
                                  default=["Under 30","30–45","46–60","Over 60"])
    st.markdown("---")
    st.markdown("**M. Anjali**  \nFinancial Analyst Intern  \nUnified Mentor × ECB  \nFebruary 2026")

# Apply filters
fdf = df[
    df["Geography"].isin(geo_filter) &
    df["Gender"].isin(gen_filter) &
    df["AgeGroup"].isin(age_filter)
]

# ── Helper functions ──────────────────────────────────────────────────────────
def kpi_card(col, value, label, card_class=""):
    col.markdown(f"""
    <div class="metric-card {card_class}">
        <h2>{value}</h2>
        <p>{label}</p>
    </div>""", unsafe_allow_html=True)

def bar_chart(ax, categories, values, colors=None, title="", ylabel="Churn Rate (%)", fmt="{:.1f}%"):
    c = colors or [LIGHT]*len(categories)
    bars = ax.bar(categories, values, color=c, edgecolor="white", linewidth=0.8, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                fmt.format(val), ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333")
    ax.set_title(title, fontsize=13, fontweight="bold", color=BLUE, pad=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(values)*1.2 if values else 1)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.markdown(f"<h1 style='color:{BLUE}'>📊 Churn Overview Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Customer Segmentation & Churn Pattern Analytics — European Banking")
    st.markdown("---")

    total      = len(fdf)
    churned    = fdf["Exited"].sum()
    retained   = total - churned
    churn_rate = fdf["Exited"].mean()*100

    c1,c2,c3,c4 = st.columns(4)
    kpi_card(c1, f"{total:,}", "Total Customers")
    kpi_card(c2, f"{churned:,}", "Churned Customers", "red-card")
    kpi_card(c3, f"{retained:,}", "Retained Customers", "green-card")
    kpi_card(c4, f"{churn_rate:.1f}%", "Overall Churn Rate", "gold-card")

    st.markdown("<div class='section-title'>Churn Distribution by Segment</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        sizes  = [retained, churned]
        colors = [GREEN, ACCENT]
        wedges, texts, autotexts = ax.pie(
            sizes, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2)
        )
        for at in autotexts:
            at.set_fontsize(12); at.set_fontweight("bold"); at.set_color("white")
        ax.legend(["Retained","Churned"], loc="lower center", ncol=2, frameon=False)
        ax.set_title("Retained vs Churned", fontsize=13, fontweight="bold", color=BLUE)
        st.pyplot(fig); plt.close()

    with col2:
        geo_churn = fdf.groupby("Geography")["Exited"].mean()*100
        fig, ax = plt.subplots(figsize=(5,4))
        geo_colors = [ACCENT if v == geo_churn.max() else LIGHT for v in geo_churn.values]
        bar_chart(ax, geo_churn.index.tolist(), geo_churn.values.tolist(),
                  geo_colors, "Churn Rate by Country")
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        age_churn = fdf.groupby("AgeGroup", observed=True)["Exited"].mean()*100
        fig, ax = plt.subplots(figsize=(5,4))
        age_colors = [ACCENT if v == age_churn.max() else LIGHT for v in age_churn.values]
        bar_chart(ax, age_churn.index.tolist(), age_churn.values.tolist(),
                  age_colors, "Churn Rate by Age Group")
        st.pyplot(fig); plt.close()

    with col4:
        prod_churn = fdf.groupby("NumOfProducts")["Exited"].mean()*100
        fig, ax = plt.subplots(figsize=(5,4))
        prod_colors = [ACCENT if v >= 50 else LIGHT for v in prod_churn.values]
        bar_chart(ax, [f"{i} Product(s)" for i in prod_churn.index],
                  prod_churn.values.tolist(), prod_colors, "Churn Rate by Products Held")
        st.pyplot(fig); plt.close()

    st.markdown("""
    <div class='insight-box'>
    💡 <b>Key Takeaway:</b> Churn is not evenly distributed. Germany, the 46–60 age group,
    and customers with 3–4 products drive the majority of exits. Targeted interventions in
    these segments will have the highest impact.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — GEOGRAPHIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Geographic Analysis":
    st.markdown(f"<h1 style='color:{BLUE}'>🌍 Geographic Churn Analysis</h1>", unsafe_allow_html=True)
    st.markdown("---")

    geo_stats = fdf.groupby("Geography").agg(
        Customers=("Exited","count"),
        Churned=("Exited","sum"),
        ChurnRate=("Exited","mean"),
        AvgBalance=("Balance","mean"),
        AvgAge=("Age","mean")
    ).reset_index()
    geo_stats["ChurnRate"] = (geo_stats["ChurnRate"]*100).round(1)
    geo_stats["AvgBalance"] = geo_stats["AvgBalance"].round(0).astype(int)
    geo_stats["AvgAge"]     = geo_stats["AvgAge"].round(1)

    c1,c2,c3 = st.columns(3)
    for col, row in zip([c1,c2,c3], geo_stats.itertuples()):
        card_class = "red-card" if row.ChurnRate == geo_stats["ChurnRate"].max() else ""
        col.markdown(f"""
        <div class='metric-card {card_class}'>
            <h2>{row.ChurnRate}%</h2>
            <p><b>{row.Geography}</b> — {row.Customers:,} customers<br>
            {row.Churned:,} churned · Avg balance €{row.AvgBalance:,}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Churn Breakdown by Country</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        bar_chart(ax, geo_stats["Geography"].tolist(), geo_stats["ChurnRate"].tolist(),
                  [ACCENT if r==geo_stats["ChurnRate"].max() else LIGHT for r in geo_stats["ChurnRate"]],
                  "Churn Rate by Country")
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        x = np.arange(len(geo_stats))
        w = 0.35
        ax.bar(x-w/2, geo_stats["Customers"]-geo_stats["Churned"], w, label="Retained", color=GREEN)
        ax.bar(x+w/2, geo_stats["Churned"], w, label="Churned", color=ACCENT)
        ax.set_xticks(x); ax.set_xticklabels(geo_stats["Geography"])
        ax.legend(); ax.set_title("Churned vs Retained by Country", fontsize=13, fontweight="bold", color=BLUE)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#FAFAFA"); ax.grid(axis="y", alpha=0.3, linestyle="--")
        st.pyplot(fig); plt.close()

    st.markdown("<div class='section-title'>Geography × Age Group Churn Heatmap</div>", unsafe_allow_html=True)
    pivot = fdf.groupby(["Geography","AgeGroup"], observed=True)["Exited"].mean()*100
    pivot = pivot.unstack("AgeGroup").round(1)
    fig, ax = plt.subplots(figsize=(10,3.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
                ax=ax, linewidths=0.5, annot_kws={"size":11,"weight":"bold"},
                cbar_kws={"label":"Churn Rate (%)"})
    ax.set_title("Churn Rate (%) — Geography × Age Group", fontsize=13, fontweight="bold", color=BLUE)
    ax.set_ylabel(""); ax.set_xlabel("")
    st.pyplot(fig); plt.close()

    st.markdown("""
    <div class='warning-box'>
    🚨 <b>Germany Alert:</b> Germany's churn rate (32.4%) is double that of France and Spain.
    German customers in the 46–60 age group churn at 67.3% — the single highest segment in the entire dataset.
    Immediate investigation is recommended.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DEMOGRAPHIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Demographic Analysis":
    st.markdown(f"<h1 style='color:{BLUE}'>👥 Demographic Churn Analysis</h1>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Churn by Age Group</div>", unsafe_allow_html=True)
        age_data = fdf.groupby("AgeGroup", observed=True).agg(
            Customers=("Exited","count"), ChurnRate=("Exited","mean")).reset_index()
        age_data["ChurnRate"] = (age_data["ChurnRate"]*100).round(1)
        fig, ax = plt.subplots(figsize=(6,4))
        age_colors = [ACCENT if v == age_data["ChurnRate"].max() else LIGHT for v in age_data["ChurnRate"]]
        bar_chart(ax, age_data["AgeGroup"].tolist(), age_data["ChurnRate"].tolist(),
                  age_colors, "Churn Rate by Age Group")
        st.pyplot(fig); plt.close()
        st.dataframe(age_data.rename(columns={"AgeGroup":"Age Group","ChurnRate":"Churn Rate (%)"}),
                     use_container_width=True, hide_index=True)

    with col2:
        st.markdown("<div class='section-title'>Churn by Gender</div>", unsafe_allow_html=True)
        gen_data = fdf.groupby("Gender").agg(
            Customers=("Exited","count"), Churned=("Exited","sum"),
            ChurnRate=("Exited","mean")).reset_index()
        gen_data["ChurnRate"] = (gen_data["ChurnRate"]*100).round(1)
        fig, ax = plt.subplots(figsize=(6,4))
        bar_chart(ax, gen_data["Gender"].tolist(), gen_data["ChurnRate"].tolist(),
                  [ACCENT, LIGHT], "Churn Rate by Gender")
        st.pyplot(fig); plt.close()
        st.dataframe(gen_data.rename(columns={"ChurnRate":"Churn Rate (%)"}),
                     use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Active vs Inactive Members</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        act_data = fdf.groupby("IsActiveMember").agg(
            Customers=("Exited","count"), Churned=("Exited","sum"),
            ChurnRate=("Exited","mean")).reset_index()
        act_data["Label"] = act_data["IsActiveMember"].map({0:"Inactive",1:"Active"})
        act_data["ChurnRate"] = (act_data["ChurnRate"]*100).round(1)
        fig, ax = plt.subplots(figsize=(5,4))
        bar_chart(ax, act_data["Label"].tolist(), act_data["ChurnRate"].tolist(),
                  [ACCENT, GREEN], "Churn Rate: Active vs Inactive")
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown("<div class='section-title'>Gender × Geography Churn</div>", unsafe_allow_html=True)
        gg = fdf.groupby(["Geography","Gender"])["Exited"].mean()*100
        gg = gg.unstack("Gender").round(1)
        fig, ax = plt.subplots(figsize=(5,4))
        x = np.arange(len(gg)); w = 0.35
        ax.bar(x-w/2, gg["Female"], w, label="Female", color=ACCENT)
        ax.bar(x+w/2, gg["Male"],   w, label="Male",   color=LIGHT)
        ax.set_xticks(x); ax.set_xticklabels(gg.index)
        ax.legend(); ax.set_title("Churn Rate by Country & Gender", fontsize=12, fontweight="bold", color=BLUE)
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#FAFAFA"); ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig); plt.close()

    st.markdown("""
    <div class='insight-box'>
    💡 <b>Demographic Insight:</b> The 46–60 age group churns at 51.1% — more than double the overall average.
    Female customers churn 8.6 percentage points higher than males across all geographies.
    Inactive members are nearly twice as likely to churn as active members.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FINANCIAL SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Financial Segmentation":
    st.markdown(f"<h1 style='color:{BLUE}'>💰 Financial Segmentation Analysis</h1>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Churn by Number of Products</div>", unsafe_allow_html=True)
        prod_data = fdf.groupby("NumOfProducts").agg(
            Customers=("Exited","count"), Churned=("Exited","sum"),
            ChurnRate=("Exited","mean")).reset_index()
        prod_data["ChurnRate"] = (prod_data["ChurnRate"]*100).round(1)
        fig, ax = plt.subplots(figsize=(6,4))
        prod_colors = [ACCENT if v >= 50 else (GREEN if v < 15 else GOLD) for v in prod_data["ChurnRate"]]
        bar_chart(ax, [f"{p} Products" for p in prod_data["NumOfProducts"]],
                  prod_data["ChurnRate"].tolist(), prod_colors, "Churn Rate by Products Held")
        st.pyplot(fig); plt.close()
        st.dataframe(prod_data.rename(columns={"NumOfProducts":"Products","ChurnRate":"Churn Rate (%)"}),
                     use_container_width=True, hide_index=True)

    with col2:
        st.markdown("<div class='section-title'>Churn by Balance Segment</div>", unsafe_allow_html=True)
        bal_data = fdf.groupby("BalanceSeg", observed=True).agg(
            Customers=("Exited","count"), ChurnRate=("Exited","mean")).reset_index()
        bal_data["ChurnRate"] = (bal_data["ChurnRate"]*100).round(1)
        fig, ax = plt.subplots(figsize=(6,4))
        bar_chart(ax, bal_data["BalanceSeg"].tolist(), bal_data["ChurnRate"].tolist(),
                  [ACCENT if v == bal_data["ChurnRate"].max() else LIGHT for v in bal_data["ChurnRate"]],
                  "Churn Rate by Balance Segment")
        st.pyplot(fig); plt.close()
        st.dataframe(bal_data.rename(columns={"BalanceSeg":"Balance Segment","ChurnRate":"Churn Rate (%)"}),
                     use_container_width=True, hide_index=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='section-title'>Churn by Credit Score Band</div>", unsafe_allow_html=True)
        cr_data = fdf.groupby("CreditBand", observed=True).agg(
            Customers=("Exited","count"), ChurnRate=("Exited","mean")).reset_index()
        cr_data["ChurnRate"] = (cr_data["ChurnRate"]*100).round(1)
        fig, ax = plt.subplots(figsize=(5,4))
        bar_chart(ax, cr_data["CreditBand"].tolist(), cr_data["ChurnRate"].tolist(),
                  [LIGHT]*len(cr_data), "Churn Rate by Credit Score Band")
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown("<div class='section-title'>High-Value Customer Churn</div>", unsafe_allow_html=True)
        threshold = fdf["Balance"].quantile(0.75)
        hv = fdf[fdf["Balance"] >= threshold]
        hv_churn  = hv["Exited"].mean()*100
        all_churn = fdf["Exited"].mean()*100
        fig, ax = plt.subplots(figsize=(5,4))
        bar_chart(ax, ["All Customers","High-Value (Top 25%)"],
                  [round(all_churn,1), round(hv_churn,1)],
                  [LIGHT, ACCENT], "High-Value vs Overall Churn Rate")
        st.pyplot(fig); plt.close()

    hv_churned = hv[hv["Exited"]==1]
    total_assets_lost = hv_churned["Balance"].sum()

    st.markdown(f"""
    <div class='warning-box'>
    🚨 <b>High-Value Risk:</b> The top 25% of customers by balance churn at {hv_churn:.1f}% — above the overall average.
    Approximately <b>€{total_assets_lost:,.0f}</b> in assets have left with churned high-value customers.
    Cross-selling 3–4 products is backfiring: 100% of 4-product customers have churned.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SEGMENT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Segment Explorer":
    st.markdown(f"<h1 style='color:{BLUE}'>🔍 Custom Segment Explorer</h1>", unsafe_allow_html=True)
    st.markdown("Compare churn rates across any two dimensions interactively.")
    st.markdown("---")

    dim_options = {
        "Geography":       "Geography",
        "Age Group":       "AgeGroup",
        "Gender":          "Gender",
        "Credit Band":     "CreditBand",
        "Tenure Group":    "TenureGroup",
        "Balance Segment": "BalanceSeg",
        "Products Held":   "NumOfProducts",
        "Active Member":   "IsActiveMember",
    }

    col1, col2 = st.columns(2)
    with col1:
        dim1 = st.selectbox("Primary Dimension", list(dim_options.keys()), index=0)
    with col2:
        dim2 = st.selectbox("Secondary Dimension (for heatmap)", list(dim_options.keys()), index=1)

    col_a, col_b = st.columns(2)

    with col_a:
        seg1 = fdf.groupby(dim_options[dim1], observed=True)["Exited"].mean()*100
        seg1 = seg1.round(1).reset_index()
        seg1.columns = [dim1, "ChurnRate"]
        fig, ax = plt.subplots(figsize=(6,4.5))
        colors = [ACCENT if v == seg1["ChurnRate"].max() else LIGHT for v in seg1["ChurnRate"]]
        bar_chart(ax, seg1[dim1].astype(str).tolist(), seg1["ChurnRate"].tolist(),
                  colors, f"Churn Rate by {dim1}")
        st.pyplot(fig); plt.close()

        seg1_full = fdf.groupby(dim_options[dim1], observed=True).agg(
            Customers=("Exited","count"), Churned=("Exited","sum"),
            ChurnRate=("Exited","mean")).reset_index()
        seg1_full["ChurnRate"] = (seg1_full["ChurnRate"]*100).round(1)
        seg1_full.columns = [dim1,"Customers","Churned","Churn Rate (%)"]
        st.dataframe(seg1_full, use_container_width=True, hide_index=True)

    with col_b:
        try:
            pivot = fdf.groupby([dim_options[dim1], dim_options[dim2]], observed=True)["Exited"].mean()*100
            pivot = pivot.unstack().round(1)
            fig, ax = plt.subplots(figsize=(6,4.5))
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
                        ax=ax, linewidths=0.5, annot_kws={"size":10,"weight":"bold"},
                        cbar_kws={"label":"Churn Rate (%)"})
            ax.set_title(f"Churn Rate: {dim1} × {dim2}", fontsize=12, fontweight="bold", color=BLUE)
            ax.set_ylabel(dim1); ax.set_xlabel(dim2)
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig); plt.close()
        except Exception:
            st.info("Select two different dimensions to view the heatmap.")

    st.markdown("<div class='section-title'>Segment Summary</div>", unsafe_allow_html=True)
    total     = len(fdf)
    churned   = fdf["Exited"].sum()
    churn_pct = fdf["Exited"].mean()*100
    highest   = seg1.loc[seg1["ChurnRate"].idxmax(), dim1]
    high_rate = seg1["ChurnRate"].max()

    c1,c2,c3 = st.columns(3)
    kpi_card(c1, f"{total:,}", f"Customers in filtered view")
    kpi_card(c2, f"{churn_pct:.1f}%", "Filtered Churn Rate", "gold-card")
    kpi_card(c3, f"{highest} — {high_rate:.1f}%", f"Highest churn {dim1}", "red-card")
