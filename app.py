
import streamlit as st
import pandas as pd
import requests
from catboost import CatBoostClassifier

# ==========================================
# 1. إعدادات الواجهة والمفاتيح
# ==========================================
st.set_page_config(page_title="توقعات البريميرليغ الذكية", page_icon="⚽", layout="centered")
st.title("⚽ محلل مباريات الدوري الإنجليزي (النسخة الاحترافية)")
st.write("نظام آلي يعتمد على لغة الأرقام وأداء الفرق الفعلي لتحليل المباريات بدقة.")

# ⚠️ استبدل هذا النص بمفتاح الـ API الحقيقي الخاص بك

API_KEY = st.secrets["ba175d4ffb7f4b07989149f202f9e019"]

# ==========================================
# 2. محرك البيانات (بدون LabelEncoder - يعتمد على الأرقام فقط)
# ==========================================
@st.cache_resource(ttl=86400) # التحديث التلقائي كل 24 ساعة
def load_assets():
    # تحميل النموذج الأعمى
    model = CatBoostClassifier()
    model.load_model('epl_prediction_model.cbm')
    
    # قراءة التاريخ
    df_history = pd.read_csv('epl_5_seasons_raw.csv')
    
    # جلب نتائج الموسم الحالي
    current_season_url = "https://api.football-data.org/v4/competitions/2021/matches?season=2026&status=FINISHED"
    headers = {'X-Auth-Token': API_KEY}
    
    try:
        response = requests.get(current_season_url, headers=headers)
        data = response.json()
        current_matches = []
        
        if 'matches' in data:
            for match in data['matches']:
                current_matches.append({
                    "Season": 2026,
                    "Match_ID": match['id'],
                    "Date": match['utcDate'],
                    "Status": match['status'],
                    "Matchday": match.get('matchday'),
                    "HomeTeam": match['homeTeam']['name'],
                    "AwayTeam": match['awayTeam']['name'],
                    "HomeGoals": match['score']['fullTime']['home'],
                    "AwayGoals": match['score']['fullTime']['away']
                })
        
        df_current = pd.DataFrame(current_matches)
        if not df_current.empty:
            df_final = pd.concat([df_history, df_current], ignore_index=True)
        else:
            df_final = df_history
            
    except Exception as e:
        df_final = df_history 
        
    return model, df_final

# استدعاء البيانات والنموذج في الذاكرة
model, df = load_assets()

# ==========================================
# 3. الدوال المساعدة
# ==========================================
def get_form(team, dataset):
    """حساب أداء الفريق في آخر 3 مباريات لعبها"""
    games = dataset[(dataset['HomeTeam'] == team) | (dataset['AwayTeam'] == team)].tail(3)
    pts, gs, gc = 0, 0, 0
    for _, row in games.iterrows():
        is_home = (row['HomeTeam'] == team)
        team_goals = row['HomeGoals'] if is_home else row['AwayGoals']
        opp_goals = row['AwayGoals'] if is_home else row['HomeGoals']
        
        gs += team_goals
        gc += opp_goals
        if team_goals > opp_goals: pts += 3
        elif team_goals == opp_goals: pts += 1
    return pts, gs, gc

def fetch_upcoming_matches():
    """جلب مباريات الجولة القادمة"""
    url = "https://api.football-data.org/v4/competitions/2021/matches?status=SCHEDULED"
    headers = {'X-Auth-Token': API_KEY}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        return data.get('matches', [])[:10] 
    except:
        return []

# ==========================================
# 4. واجهة المستخدم (التوقع والتحليل)
# ==========================================
st.markdown("---")

if st.button("🔄 جلب مباريات الجولة القادمة وتحليلها", use_container_width=True):
    with st.spinner('جاري مزامنة البيانات وحساب الاحتمالات...'):
        upcoming_matches = fetch_upcoming_matches()
        
        if not upcoming_matches:
            st.error("⚠️ لم يتم العثور على مباريات قادمة، تأكد من صحة مفتاح الـ API واتصال الإنترنت.")
        else:
            st.success("✅ تم سحب الجولة القادمة بنجاح! إليك التحليل المنطقي:")
            
            for match in upcoming_matches:
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                match_date = match['utcDate'][:10] 
                
                try:
                    # حساب فورم كل فريق من البيانات
                    h_pts, h_gs, h_gc = get_form(home_team, df)
                    a_pts, a_gs, a_gc = get_form(away_team, df)
                    h_rest, a_rest = 7, 7 
                    
                    # مصفوفة البيانات (12 ميزة رقمية فقط كما تدرب عليها النموذج)
                    match_data = pd.DataFrame({
                        'Home_Pts_L3': [h_pts], 'Away_Pts_L3': [a_pts],
                        'Home_GS_L3': [h_gs], 'Away_GS_L3': [a_gs],
                        'Home_GC_L3': [h_gc], 'Away_GC_L3': [a_gc],
                        'Home_Rest_Days': [h_rest], 'Away_Rest_Days': [a_rest],
                        'Pts_Diff': [h_pts - a_pts], 'GS_Diff': [h_gs - a_gs],
                        'GC_Diff': [h_gc - a_gc], 'Rest_Diff': [h_rest - a_rest]
                    })
                    
                    # استخراج التوقعات
                    probs = model.predict_proba(match_data)[0]
                    home_win_prob = probs[1] * 100
                    double_chance_prob = probs[0] * 100
                    
                    # عرض النتائج في قسم قابل للطي
                    with st.expander(f"⚽ {home_team} 🆚 {away_team} | 📅 {match_date}"):
                        # عرض احتمالات الفوز
                        col1, col2 = st.columns(2)
                        col1.metric(label=f"🔥 فوز {home_team}", value=f"{home_win_prob:.1f}%")
                        col2.metric(label=f"🛡️ تعثر {home_team} (تعادل/خسارة)", value=f"{double_chance_prob:.1f}%")
                        
                        st.markdown("---")
                        st.markdown("**📊 أداء الفريقين في آخر 3 مباريات:**")
                        
                        # عرض الإحصائيات في صناديق ملونة لمنع تداخل اللغات
                        stat_col1, stat_col2 = st.columns(2)
                        
                        with stat_col1:
                            st.success(f"🏠 **المضيف: {home_team}**\n\n"
                                       f"🟢 النقاط: **{h_pts}**\n\n"
                                       f"⚽ سجل: **{h_gs}** أهداف\n\n"
                                       f"🥅 استقبل: **{h_gc}** أهداف")
                            
                        with stat_col2:
                            st.info(f"✈️ **الضيف: {away_team}**\n\n"
                                    f"🟢 النقاط: **{a_pts}**\n\n"
                                    f"⚽ سجل: **{a_gs}** أهداف\n\n"
                                    f"🥅 استقبل: **{a_gc}** أهداف")
                        
                except Exception as e:
                    st.warning(f"⚠️ حدث خطأ فني في تحليل مباراة {home_team} ضد {away_team}: {e}")