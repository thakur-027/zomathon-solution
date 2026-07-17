// ══════════════════════════════════════════════════════
//  DATA
// ══════════════════════════════════════════════════════
const MENUS = {
  north: [
    { id:'n1', name:'Butter Chicken',    cat:'main_course', price:280, veg:false, emoji:'🍗' },
    { id:'n2', name:'Paneer Butter Masala',cat:'main_course',price:240, veg:true,  emoji:'🧀' },
    { id:'n3', name:'Dal Makhani',        cat:'main_course', price:180, veg:true,  emoji:'🫘' },
    { id:'n4', name:'Chicken Biryani',    cat:'main_course', price:300, veg:false, emoji:'🍚' },
    { id:'n5', name:'Chicken Tikka',      cat:'starter',     price:260, veg:false, emoji:'🍢' },
    { id:'n6', name:'Paneer Tikka',       cat:'starter',     price:220, veg:true,  emoji:'🧡' },
    { id:'n7', name:'Samosa (2 pcs)',     cat:'starter',     price:60,  veg:true,  emoji:'🥟' },
    { id:'n8', name:'Butter Naan',        cat:'side_dish',   price:40,  veg:true,  emoji:'🫓' },
    { id:'n9', name:'Raita',              cat:'side_dish',   price:60,  veg:true,  emoji:'🥛' },
    { id:'n10',name:'Gulab Jamun',        cat:'dessert',     price:80,  veg:true,  emoji:'🍮' },
    { id:'n11',name:'Kheer',              cat:'dessert',     price:90,  veg:true,  emoji:'🍨' },
    { id:'n12',name:'Mango Lassi',        cat:'beverage',    price:80,  veg:true,  emoji:'🥭' },
    { id:'n13',name:'Sweet Lassi',        cat:'beverage',    price:60,  veg:true,  emoji:'🥛' },
  ],
  south: [
    { id:'s1', name:'Masala Dosa',        cat:'main_course', price:120, veg:true,  emoji:'🫓' },
    { id:'s2', name:'Idli (4 pcs)',       cat:'main_course', price:80,  veg:true,  emoji:'🫐' },
    { id:'s3', name:'Chettinad Chicken',  cat:'main_course', price:320, veg:false, emoji:'🍗' },
    { id:'s4', name:'Vada (2 pcs)',       cat:'starter',     price:70,  veg:true,  emoji:'🍩' },
    { id:'s5', name:'Uttapam',            cat:'starter',     price:100, veg:true,  emoji:'🥞' },
    { id:'s6', name:'Coconut Chutney',    cat:'side_dish',   price:30,  veg:true,  emoji:'🥥' },
    { id:'s7', name:'Sambar',             cat:'side_dish',   price:40,  veg:true,  emoji:'🍲' },
    { id:'s8', name:'Payasam',            cat:'dessert',     price:80,  veg:true,  emoji:'🍮' },
    { id:'s9', name:'Filter Coffee',      cat:'beverage',    price:50,  veg:true,  emoji:'☕' },
    { id:'s10',name:'Fresh Lime Soda',    cat:'beverage',    price:60,  veg:true,  emoji:'🍋' },
  ],
  italian: [
    { id:'i1', name:'Margherita Pizza',   cat:'main_course', price:320, veg:true,  emoji:'🍕' },
    { id:'i2', name:'Chicken Pizza',      cat:'main_course', price:380, veg:false, emoji:'🍕' },
    { id:'i3', name:'Pasta Arrabiata',    cat:'main_course', price:260, veg:true,  emoji:'🍝' },
    { id:'i4', name:'Garlic Bread',       cat:'starter',     price:120, veg:true,  emoji:'🥖' },
    { id:'i5', name:'Bruschetta',         cat:'starter',     price:150, veg:true,  emoji:'🍞' },
    { id:'i6', name:'Caesar Salad',       cat:'side_dish',   price:180, veg:true,  emoji:'🥗' },
    { id:'i7', name:'Cheesecake Slice',   cat:'dessert',     price:180, veg:true,  emoji:'🍰' },
    { id:'i8', name:'Tiramisu',           cat:'dessert',     price:200, veg:true,  emoji:'☕' },
    { id:'i9', name:'Cold Coffee',        cat:'beverage',    price:120, veg:true,  emoji:'🧋' },
    { id:'i10',name:'Sparkling Water',    cat:'beverage',    price:80,  veg:true,  emoji:'💧' },
  ],
  chinese: [
    { id:'c1', name:'Chicken Fried Rice', cat:'main_course', price:220, veg:false, emoji:'🍳' },
    { id:'c2', name:'Veg Hakka Noodles',  cat:'main_course', price:180, veg:true,  emoji:'🍜' },
    { id:'c3', name:'Kung Pao Chicken',   cat:'main_course', price:280, veg:false, emoji:'🍗' },
    { id:'c4', name:'Veg Spring Rolls',   cat:'starter',     price:120, veg:true,  emoji:'🥢' },
    { id:'c5', name:'Chicken Dumplings',  cat:'starter',     price:160, veg:false, emoji:'🥟' },
    { id:'c6', name:'Fried Rice (plain)', cat:'side_dish',   price:100, veg:true,  emoji:'🍚' },
    { id:'c7', name:'Hot & Sour Soup',    cat:'side_dish',   price:120, veg:true,  emoji:'🍵' },
    { id:'c8', name:'Honey Noodles',      cat:'dessert',     price:100, veg:true,  emoji:'🍯' },
    { id:'c9', name:'Mango Shake',        cat:'beverage',    price:100, veg:true,  emoji:'🥭' },
    { id:'c10',name:'Iced Tea',           cat:'beverage',    price:80,  veg:true,  emoji:'🧊' },
  ],
};

// Co-occurrence matrix (simplified) — which items are commonly ordered together
const COOCCUR = {
  main_course: { side_dish: 0.85, beverage: 0.80, dessert: 0.65, starter: 0.50 },
  starter:     { main_course: 0.90, beverage: 0.70, side_dish: 0.40, dessert: 0.30 },
  side_dish:   { main_course: 0.80, beverage: 0.60, dessert: 0.50, starter: 0.35 },
  dessert:     { beverage: 0.75, main_course: 0.40, side_dish: 0.30, starter: 0.20 },
  beverage:    { main_course: 0.70, dessert: 0.55, side_dish: 0.40, starter: 0.35 },
};

const LLM_INSIGHTS = {
  has_main_no_bev: "Cart has a hearty main course but is missing a beverage to complement it. Adding a drink will make this a complete, satisfying meal.",
  has_main_no_dessert: "A rich main course is in the cart. Completing with a dessert would round out this meal experience perfectly.",
  has_main_no_side: "Main course detected — a side dish or accompaniment would enhance the meal and balance the flavours.",
  minimal: "Cart is just getting started. Recommending popular items based on what customers like you frequently order.",
  complete: "Cart is well-rounded with multiple components. Suggesting premium add-ons to enhance the overall experience.",
  only_starter: "Starter in cart — customers who order this typically follow up with a main course, beverage, and dessert.",
  dinner_premium: "Premium dinner order detected. Recommending high-quality accompaniments that match your dining preferences.",
};

// ══════════════════════════════════════════════════════
//  STATE
// ══════════════════════════════════════════════════════
let state = {
  restaurant: 'north',
  mealTime:   'dinner',
  segment:    'mid_range',
  isVeg:      false,
  cart:       [],
};

// ══════════════════════════════════════════════════════
//  RENDER MENU
// ══════════════════════════════════════════════════════
function renderMenu() {
  const menu  = MENUS[state.restaurant];
  const grid  = document.getElementById('menu-grid');
  const cats  = [...new Set(menu.map(i => i.cat))];
  const catNames = {
    main_course:'Main Course', starter:'Starters',
    side_dish:'Sides & Breads', dessert:'Desserts', beverage:'Beverages'
  };

  grid.innerHTML = cats.map(cat => {
    const items = menu.filter(i => i.cat === cat);
    const filtered = state.isVeg ? items.filter(i => i.veg) : items;
    if (!filtered.length) return '';
    return `
      <div class="category-label">${catNames[cat] || cat}</div>
      ${filtered.map(item => {
        const inCart = state.cart.some(c => c.id === item.id);
        return `
          <div class="menu-item ${inCart ? 'in-cart' : ''}" onclick="toggleCart('${item.id}')">
            <div class="item-left">
              <div class="veg-dot ${item.veg ? 'veg' : 'nonveg'}"></div>
              <span class="item-name">${item.emoji} ${item.name}</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px">
              <span class="item-price">₹${item.price}</span>
              <button class="add-btn">${inCart ? '✓' : '+'}</button>
            </div>
          </div>`;
      }).join('')}`;
  }).join('');
}

// ══════════════════════════════════════════════════════
//  CART OPERATIONS
// ══════════════════════════════════════════════════════
function toggleCart(itemId) {
  const menu  = MENUS[state.restaurant];
  const item  = menu.find(i => i.id === itemId);
  if (!item) return;

  const idx = state.cart.findIndex(c => c.id === itemId);
  if (idx >= 0) {
    state.cart.splice(idx, 1);
  } else {
    state.cart.push(item);
  }
  renderAll();
}

function removeFromCart(itemId) {
  state.cart = state.cart.filter(c => c.id !== itemId);
  renderAll();
}

// ══════════════════════════════════════════════════════
//  RENDER CART
// ══════════════════════════════════════════════════════
function renderCart() {
  const container = document.getElementById('cart-items');
  const totalEl   = document.getElementById('cart-total');
  const totalVal  = document.getElementById('total-value');

  if (state.cart.length === 0) {
    container.innerHTML = `<div class="cart-empty"><span class="icon">🛒</span>Add items from the menu to get started</div>`;
    totalEl.style.display = 'none';
    return;
  }

  const total = state.cart.reduce((s, i) => s + i.price, 0);
  container.innerHTML = state.cart.map(item => `
    <div class="cart-item">
      <div class="cart-item-info">
        <div class="cart-item-name">${item.emoji} ${item.name}</div>
        <div class="cart-item-cat">${item.cat.replace('_', ' ')}</div>
      </div>
      <span class="cart-item-price">₹${item.price}</span>
      <button class="remove-btn" onclick="removeFromCart('${item.id}')">×</button>
    </div>`).join('');

  totalEl.style.display = 'block';
  totalVal.textContent  = '₹' + total;

  // Update completeness badges
  const cats  = state.cart.map(i => i.cat);
  const comps = ['main_course','starter','side_dish','dessert','beverage'];
  const compNames = {main_course:'Main Course', starter:'Starter', side_dish:'Side Dish', dessert:'Dessert', beverage:'Beverage'};
  document.getElementById('components-row').innerHTML = comps.map(c => `
    <span class="comp-badge ${cats.includes(c) ? 'present' : 'missing'}">${compNames[c]}</span>`).join('');
}

// ══════════════════════════════════════════════════════
//  SCORING (simulates LightGBM ranking)
// ══════════════════════════════════════════════════════
function scoreItem(item) {
  const cartCats   = state.cart.map(i => i.cat);
  const cartAvgPrice = state.cart.length
    ? state.cart.reduce((s, i) => s + i.price, 0) / state.cart.length : 200;

  let score = 0;

  // 1. Co-occurrence signal (strongest feature)
  cartCats.forEach(cc => {
    const coScore = COOCCUR[cc]?.[item.cat] || 0.1;
    score += coScore * 0.35;
  });

  // 2. Meal completeness — reward filling missing slots
  const missing = ['main_course','starter','side_dish','dessert','beverage']
    .filter(c => !cartCats.includes(c));
  if (missing.includes(item.cat)) score += 0.30;

  // 3. Price ratio
  const ratio = item.price / cartAvgPrice;
  if (ratio < 0.6) score += 0.15;
  else if (ratio < 1.2) score += 0.10;
  else score += 0.03;

  // 4. Meal time boost
  if (state.mealTime === 'dinner' || state.mealTime === 'late_night') {
    if (['dessert','beverage'].includes(item.cat)) score += 0.10;
  }
  if (state.mealTime === 'breakfast') {
    if (['beverage','side_dish'].includes(item.cat)) score += 0.12;
  }

  // 5. Segment boost
  if (state.segment === 'premium' && item.price > 150) score += 0.08;
  if (state.segment === 'budget'  && item.price < 100) score += 0.10;

  // 6. LLM category boost (meal completeness analyzer)
  const llmBoost = { beverage:0.15, dessert:0.12, side_dish:0.10, starter:0.08, main_course:0.06 };
  score += (llmBoost[item.cat] || 0.05) * (cartCats.length > 0 ? 1 : 0.5);

  // 7. Veg match
  if (state.isVeg && !item.veg) score *= 0.1;

  return Math.min(0.99, Math.max(0.05, score));
}

// ══════════════════════════════════════════════════════
//  RENDER RECOMMENDATIONS
// ══════════════════════════════════════════════════════
function renderRecommendations() {
  const list    = document.getElementById('reco-list');
  const llmBox  = document.getElementById('llm-insight');
  const llmText = document.getElementById('llm-text');
  const latency = document.getElementById('latency-display');

  if (state.cart.length === 0) {
    list.innerHTML = `<div class="reco-empty">🤖 Add items to your cart to see<br>AI-powered recommendations</div>`;
    llmBox.style.display = 'none';
    latency.textContent = '— ms';
    return;
  }

  // Simulate latency
  const ms = 45 + Math.floor(Math.random() * 80);
  latency.textContent = ms + ' ms';

  // Get candidates (items not in cart)
  const menu  = MENUS[state.restaurant];
  const cartIds = state.cart.map(i => i.id);
  let candidates = menu.filter(i => !cartIds.includes(i.id));
  if (state.isVeg) candidates = candidates.filter(i => i.veg);

  // Score & rank
  const scored = candidates.map(item => ({ ...item, score: scoreItem(item) }));
  scored.sort((a, b) => b.score - a.score);
  const topN = scored.slice(0, 8);

  // LLM insight
  const cartCats = state.cart.map(i => i.cat);
  let insight = LLM_INSIGHTS.minimal;
  if (cartCats.includes('main_course') && !cartCats.includes('beverage')) {
    insight = LLM_INSIGHTS.has_main_no_bev;
  } else if (cartCats.includes('main_course') && !cartCats.includes('dessert')) {
    insight = LLM_INSIGHTS.has_main_no_dessert;
  } else if (cartCats.includes('main_course') && !cartCats.includes('side_dish')) {
    insight = LLM_INSIGHTS.has_main_no_side;
  } else if (cartCats.includes('starter') && !cartCats.includes('main_course')) {
    insight = LLM_INSIGHTS.only_starter;
  } else if (state.segment === 'premium' && state.mealTime === 'dinner') {
    insight = LLM_INSIGHTS.dinner_premium;
  } else if (cartCats.length >= 3) {
    insight = LLM_INSIGHTS.complete;
  }
  llmBox.style.display = 'block';
  llmText.textContent  = insight;

  // Render cards
  const whyMap = {
    main_course:'Completes your meal',
    starter:    'Popular with your order',
    side_dish:  'Great accompaniment',
    dessert:    'Perfect sweet ending',
    beverage:   'Refreshing complement',
  };
  const scoreColors = ['#FF5A5F','#FF7043','#FFA726','#66BB6A','#42A5F5','#AB47BC','#26C6DA','#EC407A'];

  list.innerHTML = topN.map((item, i) => {
    const pct   = Math.round(item.score * 100);
    const color = scoreColors[i] || scoreColors[4];
    return `
      <div class="reco-card" id="reco-${item.id}"
           style="--score-color:${color}; animation-delay:${i*0.05}s"
           onclick="addRecoToCart('${item.id}')">
        <div class="reco-rank">#${i+1}</div>
        <div class="reco-info">
          <div class="reco-name">${item.emoji} ${item.name}</div>
          <div class="reco-why">${whyMap[item.cat] || 'Recommended'}</div>
        </div>
        <div class="reco-price">₹${item.price}</div>
        <div class="score-bar-wrap">
          <div class="score-label">${pct}%</div>
          <div class="score-bar">
            <div class="score-fill" style="width:${pct}%;background:${color}"></div>
          </div>
        </div>
      </div>`;
  }).join('');
}

function addRecoToCart(itemId) {
  const menu = MENUS[state.restaurant];
  const item = menu.find(i => i.id === itemId);
  if (!item || state.cart.some(c => c.id === itemId)) return;
  state.cart.push(item);
  renderAll();
  // Flash the card
  setTimeout(() => {
    const card = document.getElementById('reco-' + itemId);
    if (card) card.classList.add('added');
  }, 10);
}

// ══════════════════════════════════════════════════════
//  CONTEXT SETTERS
// ══════════════════════════════════════════════════════
function updateContext() {
  state.restaurant = document.getElementById('restaurant').value;
  state.cart = [];
  renderAll();
}

function setMealTime(t) {
  state.mealTime = t;
  document.querySelectorAll('#meal-time-btns .toggle-btn')
    .forEach(b => b.classList.toggle('active', b.textContent.toLowerCase().replace(' ','_') === t || b.onclick.toString().includes(`'${t}'`)));
  renderAll();
}

function setSegment(s) {
  state.segment = s;
  document.querySelectorAll('#segment-btns .toggle-btn')
    .forEach(b => b.classList.toggle('active', b.onclick.toString().includes(`'${s}'`)));
  renderAll();
}

function setVeg(v) {
  state.isVeg = v;
  state.cart = state.cart.filter(i => !v || i.veg);
  document.querySelectorAll('#veg-btns .toggle-btn')
    .forEach(b => b.classList.toggle('active', b.onclick.toString().includes(v)));
  renderAll();
}

function renderAll() {
  renderMenu();
  renderCart();
  renderRecommendations();
}

// Init
renderAll();
