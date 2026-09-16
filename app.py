# ============================================================
# Tesla 주가·뉴스 분석 대시보드
# ============================================================

import streamlit as st
import pandas as pd

from io import StringIO
from pathlib import Path
from xgboost import XGBClassifier


# ============================================================
# 1. 페이지 기본 설정
# ============================================================

st.set_page_config(
    page_title="테슬라 주가·뉴스 분석 대시보드",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 2. 파일 경로
# ============================================================

BASE_DIR = Path(__file__).parent

DATA_PATH = (
    BASE_DIR
    / "df_ml_final.csv"
)

IMPORTANCE_PATH = (
    BASE_DIR
    / "feature_importance.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "selected_model.json"
)

KEYWORD_PATH = (
    BASE_DIR
    / "eda_keyword_summary.csv"
)

SENTIMENT_COUNTS_PATH = (
    BASE_DIR
    / "sentiment_counts.csv"
)

SENTIMENT_DAYTYPE_PATH = (
    BASE_DIR
    / "sentiment_by_daytype.csv"
)

DAILY_EDA_PATH = (
    BASE_DIR
    / "daily_eda.csv"
)

WEEKLY_EDA_PATH = (
    BASE_DIR
    / "weekly_eda.csv"
)

MONTHLY_EDA_PATH = (
    BASE_DIR
    / "monthly_eda.csv"
)

QUARTERLY_EDA_PATH = (
    BASE_DIR
    / "quarterly_eda.csv"
)


# ============================================================
# 3. 데이터 불러오기
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


@st.cache_data
def load_importance():

    importance = pd.read_csv(
        IMPORTANCE_PATH
    )

    return importance


@st.cache_data
def load_keywords():

    keywords = pd.read_csv(
        KEYWORD_PATH
    )

    return keywords


@st.cache_data
def load_sentiment_counts():

    sentiment_counts = pd.read_csv(
        SENTIMENT_COUNTS_PATH
    )

    return sentiment_counts


@st.cache_data
def load_sentiment_daytype():

    sentiment_daytype = pd.read_csv(
        SENTIMENT_DAYTYPE_PATH
    )

    return sentiment_daytype


@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model(
        MODEL_PATH
    )

    return model


@st.cache_data
def load_daily_eda():

    daily = pd.read_csv(
        DAILY_EDA_PATH
    )

    daily["Date"] = pd.to_datetime(
        daily["Date"]
    )

    return daily



@st.cache_data
def load_weekly_eda():

    weekly = pd.read_csv(
        WEEKLY_EDA_PATH
    )

    weekly["Start_Date"] = pd.to_datetime(
        weekly["Start_Date"]
    )

    weekly["End_Date"] = pd.to_datetime(
        weekly["End_Date"]
    )

    return weekly


@st.cache_data
def load_monthly_eda():

    monthly = pd.read_csv(
        MONTHLY_EDA_PATH
    )

    monthly["Start_Date"] = pd.to_datetime(
        monthly["Start_Date"]
    )

    monthly["End_Date"] = pd.to_datetime(
        monthly["End_Date"]
    )

    return monthly


@st.cache_data
def load_quarterly_eda():

    quarterly = pd.read_csv(
        QUARTERLY_EDA_PATH
    )

    quarterly["Start_Date"] = pd.to_datetime(
        quarterly["Start_Date"]
    )

    quarterly["End_Date"] = pd.to_datetime(
        quarterly["End_Date"]
    )

    return quarterly



# ============================================================
# 4. 데이터 및 모델 로드
# ============================================================

df = load_data()

importance = load_importance()

keywords = load_keywords()

sentiment_counts = load_sentiment_counts()

sentiment_daytype = load_sentiment_daytype()

daily_eda = load_daily_eda()

weekly_eda = load_weekly_eda()

monthly_eda = load_monthly_eda()

quarterly_eda = load_quarterly_eda()

model = load_model()

# ============================================================
# 5. 최종 모델 사용 변수
# ============================================================

selected_features = [
    "Daily_Return_pct",
    "Close_to_MA5",
    "Close_to_MA10",
    "Volatility_5d",
    "Volume_change_pct"
]


# ============================================================
# 6. 최종 모델 평가 결과
# ============================================================

VAL_BALANCED_ACCURACY = 0.568182

TEST_ACCURACY = 0.491071

TEST_BALANCED_ACCURACY = 0.481356

TEST_MACRO_F1 = 0.479070

TEST_ROC_AUC = 0.533590

BASELINE_ACCURACY = 0.544643


# ============================================================
# 7. 사이드바
# ============================================================


# Saved outputs from horizon_experiment_results.zip; no fitting occurs in the app.
HORIZON_CSV = {'split_summary.csv': 'Horizon,Split,Rows,Start_Date,End_Date,Latest_Label_Date,Up_Ratio\n1,Train,513,2023-01-17,2025-01-31,2025-02-03,0.52046783625731\n1,Validation,106,2025-02-10,2025-07-14,2025-07-15,0.49056603773584906\n1,Test,108,2025-07-22,2025-12-22,2025-12-23,0.5648148148148148\n5,Train,513,2023-01-17,2025-01-31,2025-02-07,0.5165692007797271\n5,Validation,106,2025-02-10,2025-07-14,2025-07-21,0.46226415094339623\n5,Test,108,2025-07-22,2025-12-22,2025-12-30,0.6203703703703703\n', 'validation_results.csv': 'Horizon,Feature_Set,Model,Accuracy,Balanced_Accuracy,Macro_F1,ROC_AUC\n1,기술지표만,Logistic Regression,0.4811320754716981,0.4821937321937322,0.47997502452947993,0.48753561253561256\n1,기술지표만,Random Forest,0.5094339622641509,0.5113960113960114,0.5050287356321839,0.5341880341880342\n1,기술지표만,XGBoost,0.5471698113207547,0.5480769230769231,0.546524064171123,0.561965811965812\n1,기술지표+뉴스감성,Logistic Regression,0.5094339622641509,0.5081908831908832,0.5066237021124239,0.516025641025641\n1,기술지표+뉴스감성,Random Forest,0.4811320754716981,0.4843304843304843,0.46743400018269843,0.4946581196581197\n1,기술지표+뉴스감성,XGBoost,0.4811320754716981,0.4825498575498576,0.47885939036381514,0.5277777777777778\n5,기술지표만,Logistic Regression,0.5754716981132075,0.575187969924812,0.5745250200695746,0.5896885069817401\n5,기술지표만,Random Forest,0.5566037735849056,0.554779806659506,0.5546616608563512,0.6015037593984963\n5,기술지표만,XGBoost,0.5849056603773585,0.5782312925170068,0.577536231884058,0.610812746151092\n5,기술지표+뉴스감성,Logistic Regression,0.6320754716981132,0.6278195488721805,0.6280701754385964,0.619405656999642\n5,기술지표+뉴스감성,Random Forest,0.5754716981132075,0.5723236663086287,0.5723890632003585,0.585750089509488\n5,기술지표+뉴스감성,XGBoost,0.5849056603773585,0.5782312925170068,0.577536231884058,0.5782312925170068\n', 'validation_news_effect.csv': 'Horizon,Model,기술지표+뉴스감성,기술지표만,News_Effect_pp\n1,Logistic Regression,0.5081908831908832,0.4821937321937322,2.5997150997151053\n1,Random Forest,0.4843304843304843,0.5113960113960114,-2.7065527065527117\n1,XGBoost,0.4825498575498576,0.5480769230769231,-6.552706552706555\n5,Logistic Regression,0.6278195488721805,0.575187969924812,5.263157894736848\n5,Random Forest,0.5723236663086287,0.554779806659506,1.754385964912275\n5,XGBoost,0.5782312925170068,0.5782312925170068,0.0\n', 'test_results.csv': 'Horizon,Feature_Set,Model,Validation_BA,Test_N,Accuracy,Balanced_Accuracy,Macro_F1,ROC_AUC,BA_Delta_vs_Baseline_pp\n1,다수 클래스 기준 모델,DummyClassifier,,108,0.5648148148148148,0.5,0.3609467455621302,0.5,0.0\n1,기술지표만,XGBoost,0.5480769230769231,108,0.5277777777777778,0.5160446459713987,0.5157802197802197,0.5497035228461807,1.6044645971398652\n1,기술지표+뉴스감성,Logistic Regression,0.5081908831908832,108,0.6018518518518519,0.584059993024067,0.5829366861248316,0.565050575514475,8.405999302406697\n5,다수 클래스 기준 모델,DummyClassifier,,108,0.6203703703703703,0.5,0.38285714285714284,0.5,0.0\n5,기술지표만,XGBoost,0.5782312925170068,108,0.6296296296296297,0.6257735711685475,0.6190476190476191,0.6665453221696396,12.577357116854749\n5,기술지표+뉴스감성,Logistic Regression,0.6278195488721805,108,0.5833333333333334,0.5364033491081179,0.5344381645751508,0.6068438296323262,3.6403349108117933\n'}

def horizon_table(name):
    return pd.read_csv(StringIO(HORIZON_CSV[name]))

st.sidebar.title(
    "Tesla Analysis"
)

page = st.sidebar.radio(
    "메뉴",
    [
        "프로젝트 개요",
        "주가 분석",
        "뉴스 분석",
        "예측 모델 결과",
        "추가 실험: 1·5거래일",
        "분석의 한계"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "분석 기간"
)

st.sidebar.write(
    "2023.01 ~ 2025.12"
)

st.sidebar.caption(
    "기존 1거래일 배포 모델"
)

st.sidebar.write(
    "XGBoost + 주가 기술지표"
)


# ============================================================
# 8. 공통 제목
# ============================================================

st.title(
    "테슬라 주가·뉴스 분석 대시보드"
)

st.caption(
    "테슬라 주가 변동과 뉴스 감성·키워드 분석 및 "
    "1·5거래일 방향성 예측 실험"
)


# ============================================================
# 프로젝트 개요
# ============================================================

if page == "프로젝트 개요":

    st.header(
        "프로젝트 개요"
    )

    st.write(
        """
        Tesla 주가 데이터와 Tesla·Elon Musk 관련 뉴스 데이터를
        결합하여 주가 변동과 뉴스의 연관성을 분석하고,
        다음 거래일의 상승·비상승 방향을 예측한 프로젝트입니다.
        후속 실험에서는 동일한 입력 날짜로 1·5거래일 예측을 비교했습니다.
        보합은 비상승에 포함합니다.
        """
    )


    # --------------------------------------------------------
    # 모델링 데이터 요약
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        st.metric(
            "모델링 데이터 수",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "모델링 데이터 시작일",
            df["Date"]
            .min()
            .strftime("%Y-%m-%d")
        )

    with col3:

        st.metric(
            "마지막 분석 기준일",
            df["Date"]
            .max()
            .strftime("%Y-%m-%d")
        )

    with col4:

        st.metric(
            "기존 선택 모델",
            "XGBoost"
        )


    st.divider()


    # --------------------------------------------------------
    # 원본 데이터 구성
    # --------------------------------------------------------

    st.subheader(
        "데이터 구성"
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "주가 거래일 수",
            "751"
        )

    with col2:

        st.metric(
            "원본 뉴스 수",
            "2,232"
        )

    with col3:

        st.metric(
            "중복 제거 후 뉴스 수",
            "1,570"
        )


    st.info(
        "뉴스 데이터는 2025-12-29까지이며, "
        "2025-12-30 주가는 2025-12-29의 "
        "다음 거래일 상승·하락 여부를 계산하기 위해 사용했습니다."
    )

        # --------------------------------------------------------
    # 뉴스 데이터 확보 현황
    # --------------------------------------------------------

    st.subheader(
        "뉴스 데이터 확보 현황"
    )

    st.write(
        "뉴스가 거래일별로 얼마나 고르게 확보되어 있는지 확인하기 위해 "
        "전체 750거래일을 기준으로 뉴스 포함 여부를 점검했습니다."
    )


    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        st.metric(
            "전체 거래일",
            "750"
        )

    with col2:

        st.metric(
            "뉴스가 있는 거래일",
            "715"
        )

    with col3:

        st.metric(
            "뉴스 포함 거래일 비율",
            "95.33%"
        )

    with col4:

        st.metric(
            "뉴스가 없는 거래일",
            "35"
        )


    col1, col2 = st.columns(
        2
    )

    with col1:

        st.metric(
            "거래일당 평균 뉴스 수",
            "2.09"
        )

    with col2:

        st.metric(
            "거래일당 뉴스 수 중앙값",
            "2"
        )


    st.success(
        "전체 거래일의 95.33%에서 최소 1건 이상의 뉴스가 확인되어 "
        "거래일별 확보율은 높았습니다. 다만 개별 이벤트·키워드의 등장 시점은 별도로 봐야 합니다."
    )

    # --------------------------------------------------------
    # 분석 절차
    # --------------------------------------------------------

    st.subheader(
        "분석 절차"
    )

    st.markdown(
        """
        **1. 데이터 수집**  
        Tesla 주가 데이터 + Tesla·Elon Musk 관련 뉴스

        ↓

        **2. 데이터 전처리**  
        뉴스 중복 제거 → 텍스트 정제 → 주식 거래일 기준 정렬

        ↓

        **3. 뉴스 데이터 확보 현황 점검**  
        전체 거래일 중 뉴스가 존재하는 날짜의 비율과 거래일당 기사 수 확인

        ↓

        **4. 탐색적 분석**  
        급등·급락일 키워드 비교 + FinBERT 감성 분석

        ↓

        **5. 기간 단위 분석**  
        일간·주간·월간·분기 단위로 주가 수익률, 뉴스 기사 수, 뉴스 감성 비교

        ↓

        **6. 예측 모델 구축**  
        주가 기술지표·뉴스 감성·선택 키워드 조합 비교

        ↓

        **7. 최종 모델 선택 및 평가**  
        검증 데이터의 균형 정확도를 기준으로 모델을 선택한 뒤
        테스트 데이터에서 일반화 성능 확인

        **8. 추가 실험**  
        같은 입력일을 사용해 1·5거래일 예측 비교. 기술지표와 뉴스 포함 조합을 별도로 평가
        """
    )


    # --------------------------------------------------------
    # 모델링 데이터 미리보기
    # --------------------------------------------------------

    st.subheader(
        "모델링 데이터 미리보기"
    )

    st.dataframe(
        df.tail(10),
        use_container_width=True
    )


# ============================================================
# 주가 분석
# ============================================================

elif page == "주가 분석":

    st.header(
        "주가 분석"
    )

    st.write(
        "선택한 기간의 Tesla 종가, 일간 수익률, 거래량을 확인합니다."
    )


    # --------------------------------------------------------
    # 날짜 범위 선택
    # --------------------------------------------------------

    min_date = (
        df["Date"]
        .min()
        .date()
    )

    max_date = (
        df["Date"]
        .max()
        .date()
    )


    date_range = st.date_input(
        "분석 기간 선택",
        value=(
            min_date,
            max_date
        ),
        min_value=min_date,
        max_value=max_date
    )


    if len(date_range) == 2:

        start_date = pd.Timestamp(
            date_range[0]
        )

        end_date = pd.Timestamp(
            date_range[1]
        )

    else:

        start_date = pd.Timestamp(
            min_date
        )

        end_date = pd.Timestamp(
            max_date
        )


    filtered = df[
        (df["Date"] >= start_date)
        &
        (df["Date"] <= end_date)
    ].copy()


    # --------------------------------------------------------
    # 기간 요약
    # --------------------------------------------------------

    if len(filtered) > 0:

        first_close = float(
            filtered.iloc[0]["Close"]
        )

        last_close = float(
            filtered.iloc[-1]["Close"]
        )

        price_change = (
            (
                last_close
                /
                first_close
            )
            - 1
        ) * 100


        col1, col2, col3 = st.columns(
            3
        )

        with col1:

            st.metric(
                "기간 시작 종가",
                f"${first_close:,.2f}"
            )

        with col2:

            st.metric(
                "기간 마지막 종가",
                f"${last_close:,.2f}"
            )

        with col3:

            st.metric(
                "기간 등락률",
                f"{price_change:.2f}%"
            )


        # ----------------------------------------------------
        # 종가 추이
        # ----------------------------------------------------

        st.subheader(
            "Tesla 종가 추이"
        )

        price_chart = (
            filtered[
                [
                    "Date",
                    "Close"
                ]
            ]
            .set_index("Date")
        )

        st.line_chart(
            price_chart
        )


        # ----------------------------------------------------
        # 일간 수익률
        # ----------------------------------------------------

        st.subheader(
            "일간 수익률 (%)"
        )

        return_chart = (
            filtered[
                [
                    "Date",
                    "Daily_Return_pct"
                ]
            ]
            .set_index("Date")
        )

        st.line_chart(
            return_chart
        )


        # ----------------------------------------------------
        # 거래량
        # ----------------------------------------------------

        st.subheader(
            "거래량"
        )

        volume_chart = (
            filtered[
                [
                    "Date",
                    "Volume"
                ]
            ]
            .set_index("Date")
        )

        st.bar_chart(
            volume_chart
        )

    else:

        st.warning(
            "선택한 기간에 데이터가 없습니다."
        )


# ============================================================
# 뉴스 분석
# ============================================================

elif page == "뉴스 분석":

    st.header(
        "뉴스 분석"
    )

    st.write(
        """
        거래일에 정렬된 뉴스 기사 수와
        FinBERT 감성 분석 결과를 확인합니다.
        """
    )


    # --------------------------------------------------------
    # 뉴스가 존재하는 거래일
    # --------------------------------------------------------

    news_days = df[
        df["article_count"] > 0
    ].copy()


    # --------------------------------------------------------
    # 뉴스 데이터 요약
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        st.metric(
            "뉴스가 존재한 거래일",
            f"{len(news_days):,}"
        )

    with col2:

        st.metric(
            "모델링 기간 뉴스 수",
            f"{int(df['article_count'].sum()):,}"
        )

    with col3:

        if len(news_days) > 0:

            avg_sentiment = (
                news_days[
                    "mean_sentiment"
                ]
                .mean()
            )

        else:

            avg_sentiment = 0

        st.metric(
            "거래일 평균 감성 점수",
            f"{avg_sentiment:.3f}"
        )

    with col4:

        st.metric(
            "FDR 5% 유의 키워드 수",
            "0"
        )


    st.divider()

    # ========================================================
    # 분석 단위 선택
    # ========================================================

    st.subheader(
        "기간 단위별 주가·뉴스 분석"
    )

    st.write(
        "일별 데이터뿐 아니라 주간·월간·분기 단위로 "
        "통합하여 주가와 뉴스의 흐름을 비교합니다."
    )

    analysis_unit = st.radio(
        "분석 단위 선택",
        [
            "일간",
            "주간",
            "월간",
            "분기"
        ],
        horizontal=True
    )


    # ========================================================
    # 선택한 기간에 맞는 데이터 준비
    # ========================================================

    if analysis_unit == "일간":

        period_view = (
            daily_eda
            .copy()
        )

        period_view[
            "기준일"
        ] = period_view[
            "Date"
        ]

        period_view[
            "수익률 (%)"
        ] = period_view[
            "Daily_Return_pct"
        ]

        period_view[
            "뉴스 기사 수"
        ] = period_view[
            "article_count"
        ]

        period_view[
            "평균 감성 점수"
        ] = period_view[
            "mean_sentiment"
        ]

        period_view[
            "긍정 뉴스 수"
        ] = period_view[
            "positive_count"
        ]

        period_view[
            "부정 뉴스 수"
        ] = period_view[
            "negative_count"
        ]


        # 뉴스가 없는 날의 감성 0은
        # 중립 감성이 아니라 "기사 없음"을 의미하므로
        # 그래프에서는 비워서 표시
        period_view.loc[
            period_view["뉴스 기사 수"] == 0,
            "평균 감성 점수"
        ] = pd.NA


    elif analysis_unit == "주간":

        period_view = (
            weekly_eda
            .copy()
        )

        period_view[
            "기준일"
        ] = period_view[
            "End_Date"
        ]

        period_view[
            "수익률 (%)"
        ] = period_view[
            "Period_Return_pct"
        ]

        period_view[
            "뉴스 기사 수"
        ] = period_view[
            "article_count"
        ]

        period_view[
            "평균 감성 점수"
        ] = period_view[
            "Mean_Sentiment"
        ]

        period_view[
            "긍정 뉴스 수"
        ] = period_view[
            "positive_count"
        ]

        period_view[
            "부정 뉴스 수"
        ] = period_view[
            "negative_count"
        ]


    elif analysis_unit == "월간":

        period_view = (
            monthly_eda
            .copy()
        )

        period_view[
            "기준일"
        ] = period_view[
            "End_Date"
        ]

        period_view[
            "수익률 (%)"
        ] = period_view[
            "Period_Return_pct"
        ]

        period_view[
            "뉴스 기사 수"
        ] = period_view[
            "article_count"
        ]

        period_view[
            "평균 감성 점수"
        ] = period_view[
            "Mean_Sentiment"
        ]

        period_view[
            "긍정 뉴스 수"
        ] = period_view[
            "positive_count"
        ]

        period_view[
            "부정 뉴스 수"
        ] = period_view[
            "negative_count"
        ]


    else:

        period_view = (
            quarterly_eda
            .copy()
        )

        period_view[
            "기준일"
        ] = period_view[
            "End_Date"
        ]

        period_view[
            "수익률 (%)"
        ] = period_view[
            "Period_Return_pct"
        ]

        period_view[
            "뉴스 기사 수"
        ] = period_view[
            "article_count"
        ]

        period_view[
            "평균 감성 점수"
        ] = period_view[
            "Mean_Sentiment"
        ]

        period_view[
            "긍정 뉴스 수"
        ] = period_view[
            "positive_count"
        ]

        period_view[
            "부정 뉴스 수"
        ] = period_view[
            "negative_count"
        ]


    # ========================================================
    # 선택 기간 요약
    # ========================================================

    period_count = len(
        period_view
    )

    avg_articles = (
        period_view[
            "뉴스 기사 수"
        ].mean()
    )

    zero_news_periods = (
        period_view[
            "뉴스 기사 수"
        ] == 0
    ).sum()


    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "집계 구간 수",
            f"{period_count:,}"
        )

    with col2:

        st.metric(
            "구간당 평균 뉴스 수",
            f"{avg_articles:.2f}"
        )

    with col3:

        st.metric(
            "뉴스 0건 구간",
            f"{zero_news_periods:,}"
        )


    if analysis_unit == "분기":

        st.info(
            "분기 데이터는 총 12개 구간으로 표본 수가 적기 때문에 "
            "통계적 결론보다는 장기적인 흐름을 확인하는 "
            "탐색적 분석으로 해석합니다."
        )


    # ========================================================
    # Tesla 기간 수익률
    # ========================================================

    st.subheader(
        f"{analysis_unit} Tesla 수익률"
    )

    return_chart = (
        period_view[
            [
                "기준일",
                "수익률 (%)"
            ]
        ]
        .set_index(
            "기준일"
        )
    )

    st.line_chart(
        return_chart
    )


        # ========================================================
    # 뉴스 기사 수
    # ========================================================

    if analysis_unit == "일간":

        st.subheader(
            "일간 뉴스 기사 수 추이"
        )

        daily_news_trend = (
            period_view[
                [
                    "기준일",
                    "뉴스 기사 수"
                ]
            ]
            .copy()
        )

        # 일별 기사 수는 0~3건 수준으로 변동이 작고
        # 750개 막대를 한 번에 표시하면 가독성이 떨어지므로
        # 20거래일 이동평균으로 흐름을 확인
        daily_news_trend[
            "20거래일 평균 뉴스 수"
        ] = (
            daily_news_trend[
                "뉴스 기사 수"
            ]
            .rolling(
                window=20,
                min_periods=1
            )
            .mean()
        )

        daily_news_trend = (
            daily_news_trend[
                [
                    "기준일",
                    "20거래일 평균 뉴스 수"
                ]
            ]
            .set_index(
                "기준일"
            )
        )

        st.line_chart(
            daily_news_trend
        )

        st.caption(
            "일별 기사 수는 750개 거래일에 걸쳐 0~수 건 수준으로 "
            "분포하므로, 전체 흐름을 보기 위해 20거래일 이동평균을 표시합니다."
        )


    else:

        st.subheader(
            f"{analysis_unit} 뉴스 기사 수"
        )

        article_chart = (
            period_view[
                [
                    "기준일",
                    "뉴스 기사 수"
                ]
            ]
            .set_index(
                "기준일"
            )
        )

        st.bar_chart(
            article_chart
        )


    # ========================================================
    # 뉴스 평균 감성
    # ========================================================

    st.subheader(
        f"{analysis_unit} FinBERT 평균 감성 점수"
    )

    sentiment_chart = (
        period_view[
            [
                "기준일",
                "평균 감성 점수"
            ]
        ]
        .set_index(
            "기준일"
        )
    )

    st.line_chart(
        sentiment_chart
    )


    # ========================================================
    # 긍정 / 부정 뉴스
    # ========================================================

    st.subheader(
        f"{analysis_unit} 긍정·부정 뉴스 추이"
    )

    sentiment_count_chart = (
        period_view[
            [
                "기준일",
                "긍정 뉴스 수",
                "부정 뉴스 수"
            ]
        ]
        .set_index(
            "기준일"
        )
    )

    st.line_chart(
        sentiment_count_chart
    )


    st.caption(
        "주간·월간·분기 감성 점수는 해당 기간의 기사 수를 반영한 "
        "가중 평균입니다. 기간 단위 분석은 뉴스와 주가의 흐름을 "
        "탐색하기 위한 것이며 인과관계를 의미하지 않습니다."
    )
   


    # --------------------------------------------------------
    # 전체 FinBERT 감성 분포
    # --------------------------------------------------------

    st.subheader(
        "전체 FinBERT 감성 분포"
    )

    sentiment_display = (
        sentiment_counts
        .copy()
    )

    sentiment_display[
        "Sentiment"
    ] = sentiment_display[
        "Sentiment"
    ].replace(
        {
            "positive": "긍정",
            "neutral": "중립",
            "negative": "부정"
        }
    )

    sentiment_display.columns = [
        "감성",
        "기사 수"
    ]

    st.dataframe(
        sentiment_display,
        use_container_width=True,
        hide_index=True
    )

    sentiment_chart = (
        sentiment_display
        .set_index("감성")
    )

    st.bar_chart(
        sentiment_chart
    )

    st.caption(
        "중복 제거 후 1,570개 뉴스 헤드라인에 대한 "
        "FinBERT 감성 분류 결과입니다."
    )


    # --------------------------------------------------------
    # 주가 변동 그룹별 평균 감성
    # --------------------------------------------------------

    st.subheader(
        "주가 변동 그룹별 평균 감성"
    )

    sentiment_summary = (
        sentiment_daytype
        .copy()
    )

    sentiment_summary[
        "Day_Type"
    ] = sentiment_summary[
        "Day_Type"
    ].replace(
        {
            "Plunge": "급락일",
            "Normal": "평범한 날",
            "Surge": "급등일"
        }
    )

    sentiment_summary.columns = [
        "주가 변동 그룹",
        "평균 감성 점수"
    ]

    st.dataframe(
        sentiment_summary,
        use_container_width=True,
        hide_index=True
    )

    sentiment_group_chart = (
        sentiment_summary
        .set_index("주가 변동 그룹")
    )

    st.bar_chart(
        sentiment_group_chart
    )

    st.caption(
        "감성 점수가 낮을수록 평균적으로 부정적인 감성이 강합니다."
    )


    # --------------------------------------------------------
    # 키워드 분석
    # --------------------------------------------------------

    st.subheader(
        "키워드 분석 요약"
    )

    surge_keywords = (
        keywords[
            keywords["Direction"] == "Surge"
        ]
        .sort_values(
            "surge_ratio",
            ascending=False
        )
        .head(5)
        .copy()
    )

    plunge_keywords = (
        keywords[
            keywords["Direction"] == "Plunge"
        ]
        .sort_values(
            "surge_ratio",
            ascending=True
        )
        .head(5)
        .copy()
    )


    col1, col2 = st.columns(
        2
    )


    with col1:

        st.markdown(
            "#### 급등일에서 상대적으로 자주 등장한 키워드"
        )

        surge_display = surge_keywords[
            [
                "term",
                "surge_days"
            ]
        ].copy()

        surge_display.columns = [
            "키워드",
            "급등일 등장 일수"
        ]

        st.dataframe(
            surge_display,
            use_container_width=True,
            hide_index=True
        )


    with col2:

        st.markdown(
            "#### 급락일에서 상대적으로 자주 등장한 키워드"
        )

        plunge_display = plunge_keywords[
            [
                "term",
                "plunge_days"
            ]
        ].copy()

        plunge_display.columns = [
            "키워드",
            "급락일 등장 일수"
        ]

        st.dataframe(
            plunge_display,
            use_container_width=True,
            hide_index=True
        )


    st.warning(
        "Benjamini-Hochberg 방식으로 다중검정을 보정한 결과, "
        "5% 유의수준을 충족한 키워드는 없었습니다. "
        "따라서 위 키워드는 통계적으로 확정된 차이나 "
        "주가 변동의 원인이 아니라 탐색적 결과로 해석해야 합니다."
    )


# ============================================================
# 예측 모델 결과
# ============================================================

elif page == "예측 모델 결과":

    st.header(
        "기존 실험: 다음 거래일 예측"
    )

    st.write(
        """
        모델과 변수 조합은 검증 데이터의 균형 정확도를 기준으로
        선택했습니다. 테스트 데이터는 최종 모델과 변수 조합을
        확정한 이후 일반화 성능을 평가하는 데 사용했습니다.
        """
    )


    # --------------------------------------------------------
    # 최종 모델
    # --------------------------------------------------------

    st.success(
        "기존 실험의 선택 모델: XGBoost + 주가 기술지표"
    )


    # --------------------------------------------------------
    # 모델 성능
    # --------------------------------------------------------

    st.subheader(
        "기존 Test 112일 평가"
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        st.metric(
            "검증 균형 정확도",
            f"{VAL_BALANCED_ACCURACY * 100:.2f}%"
        )

    with col2:

        st.metric(
            "테스트 정확도",
            f"{TEST_ACCURACY * 100:.2f}%"
        )

    with col3:

        st.metric(
            "테스트 균형 정확도",
            f"{TEST_BALANCED_ACCURACY * 100:.2f}%"
        )

    with col4:

        st.metric(
            "테스트 ROC-AUC",
            f"{TEST_ROC_AUC:.4f}"
        )


    col1, col2 = st.columns(
        2
    )

    with col1:

        st.metric(
            "테스트 Macro F1",
            f"{TEST_MACRO_F1:.4f}"
        )

    with col2:

        st.metric(
            "다수 클래스 기준 정확도",
            f"{BASELINE_ACCURACY * 100:.2f}%"
        )


    st.warning(
        "기존 선택 모델의 Test 정확도 49.11%와 균형 정확도 48.14%는 "
        "상승 고정 기준 모델의 정확도 54.46%, 균형 정확도 50.00%보다 낮았습니다. "
        "따라서 안정적인 일반화 예측력을 확보했다고 보기는 어렵습니다."
    )


    # --------------------------------------------------------
    # 변수 중요도
    # --------------------------------------------------------

    st.subheader(
        "변수 중요도"
    )

    importance_display = (
        importance
        .copy()
    )

    st.dataframe(
        importance_display,
        use_container_width=True,
        hide_index=True
    )

    importance_chart = (
        importance_display
        .sort_values(
            "Importance"
        )
        .set_index(
            "Feature"
        )
    )

    st.bar_chart(
        importance_chart
    )

    st.caption(
        "표의 변수명은 실제 모델에 사용된 변수명을 그대로 표시합니다. "
        "변수 중요도는 모델이 각 변수를 상대적으로 얼마나 활용했는지를 "
        "나타내며 인과관계를 의미하지 않습니다."
    )


    # --------------------------------------------------------
    # 과거 테스트 데이터 예측 확인
    # --------------------------------------------------------

    st.subheader(
        "과거 데이터 예측 확인"
    )

    st.info(
        "아래 기능은 실시간 미래 주가 예측이 아닙니다. "
        "이미 결과를 알고 있는 과거 테스트 기간에 "
        "저장된 최종 모델을 적용하여 예측 결과를 확인하는 기능입니다."
    )


    test_df = (
        df.iloc[-112:]
        .copy()
        .reset_index(drop=True)
    )


    date_options = (
        test_df["Date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
        .tolist()
    )


    selected_date = st.selectbox(
        "테스트 기준일 선택",
        options=date_options,
        index=len(date_options) - 1
    )


    selected_row = (
        test_df[
            test_df[
                "Date"
            ].dt.strftime(
                "%Y-%m-%d"
            )
            == selected_date
        ]
        .iloc[0]
    )


    X_selected = pd.DataFrame(
        [
            selected_row[
                selected_features
            ].to_dict()
        ]
    )


    predicted_class = int(
        model.predict(
            X_selected
        )[0]
    )


    predicted_probability = float(
        model.predict_proba(
            X_selected
        )[0][1]
    )


    actual_class = int(
        selected_row[
            "Target"
        ]
    )


    if predicted_class == 1:

        predicted_text = "상승"

    else:

        predicted_text = "비상승"


    if actual_class == 1:

        actual_text = "상승"

    else:

        actual_text = "비상승"


    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "모델 예측",
            predicted_text
        )

    with col2:

        st.metric(
            "상승 예측 확률",
            f"{predicted_probability * 100:.1f}%"
        )

    with col3:

        st.metric(
            "실제 방향",
            actual_text
        )


    if predicted_class == actual_class:

        st.success(
            "이 관측치에서는 모델 예측과 실제 방향이 일치했습니다."
        )

    else:

        st.error(
            "이 관측치에서는 모델 예측과 실제 방향이 일치하지 않았습니다."
        )


    st.write(
        "**모델에 입력된 주가 기술지표**"
    )

    feature_display = (
        X_selected
        .T
        .reset_index()
    )

    feature_display.columns = [
        "Feature",
        "Value"
    ]

    st.dataframe(
        feature_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 분석의 한계
# ============================================================

elif page == "추가 실험: 1·5거래일":
    st.header("추가 실험: 1·5거래일 예측 비교")
    st.write(
        "예측 기간을 늘리면 뉴스 변수의 기여가 달라지는지 확인했습니다. "
        "노트북에서 실행한 후속 실험의 저장된 결과를 보여줍니다."
    )
    st.caption(
        "두 기간에 같은 입력 날짜를 사용했습니다. 기존 Test 112일과는 "
        "표본·분할 경계 처리가 달라, 이번 108일 결과를 별도로 해석합니다."
    )
    horizon = st.radio("예측 기간", [1, 5],
                       format_func=lambda h: f"{h}거래일 뒤", horizontal=True)
    results = horizon_table("test_results.csv")
    validation = horizon_table("validation_results.csv")
    splits = horizon_table("split_summary.csv")
    effects = horizon_table("validation_news_effect.csv")
    chosen = results.loc[results["Horizon"] == horizon].copy()

    st.subheader("실험 설계")
    st.write(
        "현재 종가 대비 미래 종가가 상승하면 1, 보합·하락이면 0입니다. "
        "기술지표만 사용하는 조합과 기술지표에 뉴스 감성을 더한 조합에서 "
        "각각 3개 모델을 비교했습니다. 키워드는 이번 추가 실험에서 제외했습니다."
    )
    split_view = splits.loc[splits["Horizon"] == horizon].copy()
    split_view["Up_Ratio"] = split_view["Up_Ratio"].map(lambda v: f"{v:.2%}")
    st.dataframe(split_view.drop(columns="Horizon").rename(columns={
        "Split": "구분", "Rows": "표본 수", "Start_Date": "입력 시작일",
        "End_Date": "입력 종료일", "Latest_Label_Date": "마지막 정답 기준일",
        "Up_Ratio": "상승 비율"
    }), hide_index=True, use_container_width=True)
    st.caption(
        "5거래일 정답 기간이 다음 분할에 걸치는 관측치는 제외했습니다. "
        "5거래일 실험은 매일 계산한 향후 수익률 예측이며, 주별 1행 집계와 다릅니다."
    )

    st.subheader("Validation에서 선택한 모델")
    models = chosen.loc[chosen["Model"] != "DummyClassifier"].copy()
    selection = models[["Feature_Set", "Model", "Validation_BA"]].copy()
    selection["Validation_BA"] = selection["Validation_BA"].map(lambda v: f"{v:.2%}")
    st.dataframe(selection.rename(columns={
        "Feature_Set": "변수 조합", "Model": "선택 모델",
        "Validation_BA": "Validation 균형 정확도"
    }), hide_index=True, use_container_width=True)
    st.caption("각 변수 조합의 Validation 균형 정확도 1위 모델을 Train 학습 상태로 Test 평가했습니다.")

    st.subheader("Test 성능 비교")
    display_results = chosen[["Feature_Set", "Model", "Accuracy", "Balanced_Accuracy",
                              "Macro_F1", "ROC_AUC", "BA_Delta_vs_Baseline_pp"]].copy()
    for column in ["Accuracy", "Balanced_Accuracy"]:
        display_results[column] = display_results[column].map(lambda v: f"{v:.2%}")
    for column in ["Macro_F1", "ROC_AUC"]:
        display_results[column] = display_results[column].map(lambda v: f"{v:.4f}")
    display_results["BA_Delta_vs_Baseline_pp"] = display_results["BA_Delta_vs_Baseline_pp"].map(
        lambda v: f"{v:+.2f}%p")
    st.dataframe(display_results.rename(columns={
        "Feature_Set": "변수 조합", "Model": "모델", "Accuracy": "정확도",
        "Balanced_Accuracy": "균형 정확도", "Macro_F1": "Macro F1", "ROC_AUC": "ROC-AUC",
        "BA_Delta_vs_Baseline_pp": "기준 대비 균형 정확도 차이"
    }), hide_index=True, use_container_width=True)
    st.caption(
        "기준 모델은 각 기간 Train의 다수 클래스인 상승을 모든 날에 예측합니다. "
        "정확도·균형 정확도는 %, Macro F1·ROC-AUC는 0~1로 표시합니다."
    )
    chart = chosen.set_index("Feature_Set")[["Balanced_Accuracy"]].mul(100)
    st.bar_chart(chart.rename(columns={"Balanced_Accuracy": "Test 균형 정확도 (%)"}))

    if horizon == 5:
        st.info(
            "기술지표 XGBoost의 Test 균형 정확도는 62.58%로 기준보다 12.58%p 높았습니다. "
            "일반 정확도는 62.96%로 기준보다 0.93%p 높았습니다."
        )
        st.write(
            "5거래일 전체 Validation 1위는 뉴스 포함 Logistic Regression(62.78%)이었으나, "
            "Test 균형 정확도는 53.64%로 낮아졌습니다. "
            "Test에서 더 좋았다는 이유로 최종 모델을 XGBoost로 재선택하지 않았습니다."
        )
    else:
        st.info(
            "뉴스 포함 Logistic Regression의 Test 균형 정확도는 58.41%, "
            "기술지표 XGBoost는 51.60%였습니다. "
            "두 변수 조합은 알고리즘도 달라, 차이를 뉴스만의 효과로 해석할 수 없습니다."
        )

    st.subheader("같은 모델에서 뉴스 변수 추가 효과")
    effect_view = effects.loc[effects["Horizon"] == horizon].copy()
    for column in ["기술지표만", "기술지표+뉴스감성"]:
        effect_view[column] = effect_view[column].map(lambda v: f"{v:.2%}")
    effect_view["News_Effect_pp"] = effect_view["News_Effect_pp"].map(lambda v: f"{v:+.2f}%p")
    st.dataframe(effect_view.drop(columns="Horizon").rename(columns={
        "Model": "모델", "News_Effect_pp": "뉴스 추가 효과"
    }), hide_index=True, use_container_width=True)
    st.caption("Validation 균형 정확도 비교입니다. 양수는 뉴스 추가 후 개선, 음수는 하락을 뜻합니다.")

    with st.expander("전체 Validation 결과"):
        val_view = validation.loc[validation["Horizon"] == horizon].copy()
        for column in ["Accuracy", "Balanced_Accuracy"]:
            val_view[column] = val_view[column].map(lambda v: f"{v:.2%}")
        for column in ["Macro_F1", "ROC_AUC"]:
            val_view[column] = val_view[column].map(lambda v: f"{v:.4f}")
        st.dataframe(val_view.drop(columns="Horizon"), hide_index=True, use_container_width=True)

    st.subheader("해석 범위")
    st.write(
        "기존 Test를 본 뒤 설계한 후속 탐색 실험이며, 새 독립 표본의 검증은 아닙니다. "
        "5거래일 수익률은 인접 날짜끼리 기간이 겹칩니다. "
        "개선의 지속성과 뉴스 변수의 기여는 새 기간에서 추가 검증해야 합니다."
    )
    st.download_button(
        "전체 Test 결과 CSV 다운로드",
        data=HORIZON_CSV["test_results.csv"].encode("utf-8-sig"),
        file_name="horizon_test_results.csv", mime="text/csv"
    )


elif page == "분석의 한계":

    st.header(
        "분석의 한계 및 결론"
    )


    # --------------------------------------------------------
    # 분석의 주요 한계
    # --------------------------------------------------------

    st.subheader(
    "분석의 주요 한계"
    )

    st.markdown(
    """
    **1. 뉴스 발행 시각의 한계**  
    기사별 정확한 발행 시각이 없어 뉴스가 주식시장 마감 이전에 공개되었는지를 엄밀하게 구분할 수 없습니다.

    **2. 연관성과 인과관계의 구분**  
    뉴스 감성이나 특정 키워드가 주가 변동일에 함께 나타났더라도, 해당 뉴스가 주가 변동을 직접 일으켰다고 해석할 수는 없습니다.

    **3. 뉴스 및 키워드 분석의 한계**  
    전체 거래일의 95.33%에서 뉴스가 확인되었지만 거래일당 뉴스 수 중앙값은 2건이었으며, 다중검정 보정 후 통계적으로 유의한 키워드는 없었습니다.

    **4. 기간 단위 분석의 표본 수 차이**  
    일간 750개, 주간 157개, 월간 36개, 분기 12개로 분석했으며, 특히 분기 분석은 표본 수가 적어 장기 흐름을 확인하는 보조 분석으로 해석해야 합니다.

    **5. 기존 실험과 후속 실험의 구분**  
    기존 선택 모델의 Test 정확도·균형 정확도는 각각 49.11%·48.14%로, 기준의 54.46%·50.00%보다 낮았습니다. 추가 실험은 기존 Test를 확인한 뒤 설계한 탐색 분석으로, 새 독립 표본에서의 검증은 아닙니다.

    **6. 겹치는 예측 기간과 일반화 범위**  
    5거래일 수익률은 인접 날짜끼리 기간이 겹칩니다. 단일 종목과 Test 구간에서 관찰한 성능이므로 새 기간에서의 재검증이 필요합니다.
    """
    )
# --------------------------------------------------------
    # 결론
    # --------------------------------------------------------

    st.subheader(
        "결론"
    )

    st.markdown(
        """
        - **뉴스 분석:** Tesla 관련 뉴스의 키워드와 감성은 주가 급등일과 급락일 사이에서 일부 차이를 보였지만 다중검정 보정 후 통계적으로 유의한 키워드는 확인되지 않았습니다.

        - **기간 단위 분석:** 일간뿐 아니라 주간·월간·분기 단위로 데이터를 통합하여 주가 수익률, 뉴스 기사 수, 뉴스 감성의 흐름을 비교했습니다.

        - **기존 모델 선택:** 여러 모델과 변수 조합을 비교한 결과, **XGBoost와 주가 기술지표만 사용한 조합**이 검증 데이터에서 가장 높은 균형 정확도를 기록했습니다.

        - **기존 예측 성능:** 선택 모델의 Test 정확도·균형 정확도는 기준 모델보다 낮았습니다.

        - **추가 실험:** 5거래일 기술지표 XGBoost의 Test 균형 정확도는 62.58%였습니다. 다만 Validation 1위 뉴스 포함 모델은 Test에서 53.64%로 낮아져, 뉴스 변수의 안정적인 기여는 확인하지 못했습니다.
        """
    )

    st.info(
        "이 프로젝트의 핵심은 높은 주가 예측 정확도를 확보한 것이 아니라, "
        "뉴스와 주가 데이터를 다양한 시간 단위에서 결합해 분석하고 "
        "탐색적 패턴과 실제 예측 성능 사이의 차이를 확인했다는 데 있습니다."
    )