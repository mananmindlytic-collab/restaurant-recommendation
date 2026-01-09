"""
Restaurant Search Engine with LLM-Powered Recommendations
Hybrid search: Filtering + Semantic Ranking + LLM Menu Selection
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# SEARCH ENGINE
# ============================================================================

class RestaurantSearchEngine:
    """
    Hybrid search engine for restaurant recommendations
    """
    
    def __init__(self, index_path: str = "restaurants_search_index.json"):
        """Load search index and model"""
        print("🔄 Loading search index...")
        with open(index_path, 'r') as f:
            self.index = json.load(f)
        
        print(f"✅ Loaded {self.index['total_restaurants']} restaurants")
        
        print("🔄 Loading embedding model...")
        self.model = SentenceTransformer(self.index['embedding_model'].replace('sentence-transformers/', ''))
        print("✅ Model loaded!")
        
        print("🔄 Initializing LLM for recommendations...")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        print("✅ Ready to search!")
    
    
    def search(
        self, 
        user_requirements: Dict,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Main search function
        
        Args:
            user_requirements: Dict with user preferences
            top_k: Number of results to return
        
        Returns:
            List of restaurant recommendations with reasoning
        """
        print(f"\n{'='*80}")
        print(f"🔍 SEARCHING RESTAURANTS")
        print(f"{'='*80}")
        
        # Step 1: Mandatory filtering
        print("\n📊 Step 1: Filtering by mandatory requirements...")
        candidates = self._filter_mandatory(user_requirements)
        print(f"   ✓ Found {len(candidates)} restaurants matching mandatory criteria")
        
        if not candidates:
            return self._handle_no_results(user_requirements)
        
        # Step 2: Optional filtering
        print("\n📊 Step 2: Filtering by optional preferences...")
        candidates = self._filter_optional(user_requirements, candidates)
        print(f"   ✓ Narrowed to {len(candidates)} restaurants")
        
        # Step 3: Semantic ranking
        print("\n📊 Step 3: Ranking by semantic similarity...")
        ranked_results = self._semantic_rank(user_requirements, candidates, top_k)
        print(f"   ✓ Selected top {len(ranked_results)} restaurants")
        
        # Step 4: LLM-powered menu selection & reasoning
        print("\n📊 Step 4: Generating personalized recommendations...")
        final_results = self._generate_recommendations(user_requirements, ranked_results)
        print(f"   ✓ Generated recommendations with reasoning")
        
        return final_results
    
    
    def _filter_mandatory(self, reqs: Dict) -> List[Dict]:
        """Filter by mandatory requirements (guests, budget, timing) - flexible matching"""
        candidates = []
        
        print(f"   Looking for: {reqs['num_guests']} guests, ${reqs['budget']} budget, {reqs['timing']}")
        
        for restaurant in self.index['restaurants']:
            filters = restaurant['filters']
            
            # Guest capacity check - FLEXIBLE: allow if close to capacity
            guest_check = (
                filters['guest_min'] <= reqs['num_guests'] <= filters['guest_max'] or
                # Allow if within 2 guests of min
                (filters['guest_min'] - 2 <= reqs['num_guests'] < filters['guest_min']) or
                # Allow if within 5 guests of max
                (filters['guest_max'] < reqs['num_guests'] <= filters['guest_max'] + 5)
            )
            
            if not guest_check:
                continue
            
            # Budget check - FLEXIBLE: allow if within 30% over budget
            budget_check = (
                reqs['budget'] >= filters['price_min'] or
                reqs['budget'] >= filters['price_min'] * 0.7  # Allow if budget is 70% of min price
            )
            
            if not budget_check:
                continue
            
            # Timing check - STRICT (must match meal service)
            if reqs['timing'] not in filters['meal_services']:
                continue
            
            # Passed checks
            print(f"   ✓ {restaurant['name']}: guests({filters['guest_min']}-{filters['guest_max']}), price(${filters['price_min']}-${filters['price_max']}), timing({filters['meal_services']})")
            candidates.append(restaurant)
        
        return candidates
    
    
    def _filter_optional(self, reqs: Dict, candidates: List[Dict]) -> List[Dict]:
        """Filter by optional requirements (cuisine, dietary) - with scoring for partial matches"""
        
        # If no optional filters provided, return all candidates
        if not reqs.get('cuisine') and not reqs.get('dietary'):
            return candidates
        
        # Score each candidate based on optional matches
        scored_candidates = []
        
        for restaurant in candidates:
            score = 0
            
            # Cuisine match (if provided)
            if reqs.get('cuisine'):
                cuisine_lower = reqs['cuisine'].lower()
                restaurant_cuisines = [c.lower() for c in restaurant['filters']['cuisines']]
                
                # Exact match
                if cuisine_lower in restaurant_cuisines:
                    score += 2
                # Partial match (e.g., "italian" in "new italian")
                elif any(cuisine_lower in c for c in restaurant_cuisines):
                    score += 1
            
            # Dietary match (if provided)
            if reqs.get('dietary'):
                dietary_lower = reqs['dietary'].lower()
                restaurant_dietary = [d.lower() for d in restaurant['filters']['dietary_options']]
                
                # Exact match
                if dietary_lower in restaurant_dietary:
                    score += 2
                # Partial match
                elif any(dietary_lower in d for d in restaurant_dietary):
                    score += 1
            
            scored_candidates.append({
                'restaurant': restaurant,
                'optional_score': score
            })
        
        # Sort by optional score (highest first)
        scored_candidates.sort(key=lambda x: x['optional_score'], reverse=True)
        
        # If we have good matches (score > 0), prefer them
        good_matches = [c['restaurant'] for c in scored_candidates if c['optional_score'] > 0]
        
        if good_matches:
            print(f"   ✓ Found {len(good_matches)} with optional preferences")
            return good_matches
        
        # If no matches for optional, return all candidates (don't filter out)
        print(f"   ℹ️  No exact optional matches, showing all {len(candidates)} candidates")
        return candidates
    
    
    def _semantic_rank(
        self, 
        reqs: Dict, 
        candidates: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Rank candidates by semantic similarity"""
        
        # Build query text from vibe and occasion
        query_parts = []
        if reqs.get('vibe'):
            query_parts.append(reqs['vibe'])
        if reqs.get('occasion'):
            query_parts.append(reqs['occasion'])
        
        # If no vibe/occasion, use general query
        if not query_parts:
            query_text = f"great {reqs['timing']} restaurant"
        else:
            query_text = " ".join(query_parts)
        
        print(f"   Query: '{query_text}'")
        
        # Generate query embedding
        query_embedding = self.model.encode([query_text])[0]
        
        # Calculate similarity for each candidate
        results = []
        for candidate in candidates:
            restaurant_embedding = np.array(candidate['embedding'])
            
            # Cosine similarity
            similarity = np.dot(query_embedding, restaurant_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(restaurant_embedding)
            )
            
            results.append({
                'restaurant': candidate,
                'similarity_score': float(similarity)
            })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top K
        return results[:top_k]
    
    
    def _generate_recommendations(
        self,
        reqs: Dict,
        ranked_results: List[Dict]
    ) -> List[Dict]:
        """
        Use LLM to select best menu and generate reasoning for each restaurant
        """
        recommendations = []
        
        for i, result in enumerate(ranked_results):
            restaurant = result['restaurant']
            similarity_score = result['similarity_score']
            
            print(f"\n   Analyzing {restaurant['name']}...")
            
            # Get recommendation from LLM
            recommendation = self._get_llm_recommendation(
                reqs, 
                restaurant,
                similarity_score
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    
    def _get_llm_recommendation(
        self,
        reqs: Dict,
        restaurant: Dict,
        similarity_score: float
    ) -> Dict:
        """
        Use LLM to select specific menu and generate detailed reasoning
        """
        
        # Extract restaurant data
        full_data = restaurant['full_data']
        menus = full_data.get('menus', [])
        
        # Build prompt for LLM
        prompt = f"""
You are an expert restaurant concierge helping a guest find the perfect dining experience.

GUEST REQUIREMENTS:
- Party size: {reqs['num_guests']} guests
- Budget: ${reqs['budget']} per person
- Meal: {reqs['timing'].title()}
- Cuisine preference: {reqs.get('cuisine', 'Any')}
- Vibe: {reqs.get('vibe', 'Not specified')}
- Occasion: {reqs.get('occasion', 'Not specified')}
- Dietary needs: {reqs.get('dietary', 'None')}

RESTAURANT: {restaurant['name']}
Price Range: ${restaurant['filters']['price_min']}-${restaurant['filters']['price_max']} per person
Cuisines: {', '.join(restaurant['filters']['cuisines'])}
Vibes: {', '.join(restaurant['filters']['vibes'])}
Occasions: {', '.join(restaurant['filters']['occasions'])}

AVAILABLE MENUS:
{json.dumps([{
    'name': m['menuName'],
    'price': m['foodPrice'],
    'description': m['menuDescription'],
    'highlights': m.get('highlights', [])[:5],
    'vibe': m.get('vibe', []),
    'summary': m.get('summary', '')
} for m in menus], indent=2)}

YOUR TASK:
1. Select the BEST menu for this guest (consider budget, vibe, occasion)
2. Explain in 2-3 sentences WHY this restaurant and menu are perfect for them
3. Highlight 2-3 specific dishes that match their preferences

Return ONLY a JSON object:
{{
  "recommended_menu": "menu name",
  "menu_price": price,
  "reasoning": "2-3 sentence explanation why this is perfect",
  "dish_highlights": ["dish 1", "dish 2", "dish 3"],
  "perfect_for": "One sentence about what makes this ideal for their occasion/vibe"
}}

Be enthusiastic but concise. Focus on what makes THIS choice special for THEIR needs.
"""
        
        # Call LLM
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a restaurant expert. Return only valid JSON."),
                HumanMessage(content=prompt)
            ])
            
            # Parse response
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            llm_rec = json.loads(content)
            
            # Build final recommendation
            return {
                'rank': len([r for r in [] if r]) + 1,  # Will be set by caller
                'restaurant_name': restaurant['name'],
                'restaurant_id': restaurant['restaurant_id'],
                'cuisines': restaurant['filters']['cuisines'],
                'price_range': {
                    'min': restaurant['filters']['price_min'],
                    'max': restaurant['filters']['price_max']
                },
                'recommended_menu': llm_rec['recommended_menu'],
                'menu_price': llm_rec['menu_price'],
                'reasoning': llm_rec['reasoning'],
                'dish_highlights': llm_rec['dish_highlights'],
                'perfect_for': llm_rec['perfect_for'],
                'vibes': restaurant['filters']['vibes'],
                'occasions': restaurant['filters']['occasions'],
                'similarity_score': similarity_score,
                'full_restaurant_data': full_data
            }
        
        except Exception as e:
            print(f"   ⚠️  LLM error: {e}")
            
            # Fallback: Simple recommendation without LLM
            best_menu = menus[0] if menus else {}
            return {
                'rank': 1,
                'restaurant_name': restaurant['name'],
                'restaurant_id': restaurant['restaurant_id'],
                'cuisines': restaurant['filters']['cuisines'],
                'price_range': {
                    'min': restaurant['filters']['price_min'],
                    'max': restaurant['filters']['price_max']
                },
                'recommended_menu': best_menu.get('menuName', 'Main Menu'),
                'menu_price': best_menu.get('foodPrice', restaurant['filters']['price_min']),
                'reasoning': f"Great match for {reqs['timing']} with {reqs['num_guests']} guests.",
                'dish_highlights': best_menu.get('highlights', [])[:3],
                'perfect_for': f"Perfect for {reqs.get('occasion', 'your group')}",
                'vibes': restaurant['filters']['vibes'],
                'occasions': restaurant['filters']['occasions'],
                'similarity_score': similarity_score,
                'full_restaurant_data': full_data
            }
    
    
    def _handle_no_results(self, reqs: Dict) -> List[Dict]:
        """Handle case when no restaurants match - try different relaxation strategies"""
        print("\n   ⚠️  No exact matches found")
        print("   💡 Trying alternative search strategies...")
        
        # Strategy 1: Ignore timing, just match on guests and budget
        print("\n   Strategy 1: Relaxing meal timing...")
        candidates = []
        for restaurant in self.index['restaurants']:
            filters = restaurant['filters']
            
            # Just check guests and budget
            guest_ok = filters['guest_min'] <= reqs['num_guests'] <= filters['guest_max'] + 5
            budget_ok = reqs['budget'] >= filters['price_min'] * 0.7
            
            if guest_ok and budget_ok:
                candidates.append(restaurant)
        
        if candidates:
            print(f"   ✓ Found {len(candidates)} restaurants (relaxed timing)")
            # Continue with semantic ranking
            ranked = self._semantic_rank(reqs, candidates, 3)
            return self._generate_recommendations(reqs, ranked)
        
        # Strategy 2: Just match timing, ignore guest capacity
        print("\n   Strategy 2: Relaxing guest capacity...")
        candidates = []
        for restaurant in self.index['restaurants']:
            filters = restaurant['filters']
            
            if reqs['timing'] in filters['meal_services']:
                candidates.append(restaurant)
        
        if candidates:
            print(f"   ✓ Found {len(candidates)} restaurants for {reqs['timing']}")
            # Sort by how close they are to budget
            candidates_sorted = sorted(
                candidates, 
                key=lambda r: abs(r['filters']['price_min'] - reqs['budget'])
            )
            ranked = candidates_sorted[:3]
            
            # Create simple results
            results = []
            for i, restaurant in enumerate(ranked):
                full_data = restaurant['full_data']
                menus = full_data.get('menus', [])
                best_menu = menus[0] if menus else {}
                
                results.append({
                    'rank': i + 1,
                    'restaurant_name': restaurant['name'],
                    'restaurant_id': restaurant['restaurant_id'],
                    'cuisines': restaurant['filters']['cuisines'],
                    'price_range': {
                        'min': restaurant['filters']['price_min'],
                        'max': restaurant['filters']['price_max']
                    },
                    'recommended_menu': best_menu.get('menuName', 'Main Menu'),
                    'menu_price': best_menu.get('foodPrice', restaurant['filters']['price_min']),
                    'reasoning': f"Alternative option for {reqs['timing']} - may need to adjust group size or budget.",
                    'dish_highlights': best_menu.get('highlights', [])[:3],
                    'perfect_for': f"Flexible dining for {reqs['timing']}",
                    'vibes': restaurant['filters']['vibes'],
                    'occasions': restaurant['filters']['occasions'],
                    'similarity_score': 0.5,
                    'full_restaurant_data': full_data
                })
            
            return results
        
        # Strategy 3: Return ANY top 3 restaurants
        print("\n   Strategy 3: Showing top general recommendations...")
        if self.index['restaurants']:
            top_3 = self.index['restaurants'][:3]
            
            results = []
            for i, restaurant in enumerate(top_3):
                full_data = restaurant['full_data']
                menus = full_data.get('menus', [])
                best_menu = menus[0] if menus else {}
                
                results.append({
                    'rank': i + 1,
                    'restaurant_name': restaurant['name'],
                    'restaurant_id': restaurant['restaurant_id'],
                    'cuisines': restaurant['filters']['cuisines'],
                    'price_range': {
                        'min': restaurant['filters']['price_min'],
                        'max': restaurant['filters']['price_max']
                    },
                    'recommended_menu': best_menu.get('menuName', 'Main Menu'),
                    'menu_price': best_menu.get('foodPrice', restaurant['filters']['price_min']),
                    'reasoning': "No exact matches found. These are our popular restaurants that you might enjoy.",
                    'dish_highlights': best_menu.get('highlights', [])[:3],
                    'perfect_for': "Worth considering with flexible requirements",
                    'vibes': restaurant['filters']['vibes'],
                    'occasions': restaurant['filters']['occasions'],
                    'similarity_score': 0.3,
                    'full_restaurant_data': full_data
                })
            
            return results
        
        return []


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_recommendation(rec: Dict, rank: int) -> str:
    """Format a single recommendation for display"""
    output = f"""
{'='*80}
🏆 RECOMMENDATION #{rank}: {rec['restaurant_name']}
{'='*80}

📍 {', '.join(rec['cuisines'])}
💰 ${rec['price_range']['min']}-${rec['price_range']['max']} per person

🍽️  RECOMMENDED MENU:
{rec['recommended_menu']} - ${rec['menu_price']}/person

✨ WHY THIS RESTAURANT:
{rec['reasoning']}

🎯 PERFECT FOR:
{rec['perfect_for']}

🍴 SIGNATURE DISHES TO TRY:
"""
    
    for i, dish in enumerate(rec['dish_highlights'], 1):
        output += f"\n   {i}. {dish}"
    
    output += f"\n\n💫 Vibes: {', '.join(rec['vibes'][:4])}"
    output += f"\n🎉 Great for: {', '.join(rec['occasions'][:4])}"
    
    return output


def search_restaurants(user_requirements: Dict) -> List[Dict]:
    """
    Main function to search for restaurants
    
    Args:
        user_requirements: Dict with user preferences
    
    Returns:
        List of top 3 recommendations
    """
    engine = RestaurantSearchEngine()
    results = engine.search(user_requirements, top_k=3)
    return results


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    import os
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY")
        exit(1)
    
    print("\n" + "="*80)
    print("🍽️  RESTAURANT SEARCH ENGINE - TEST")
    print("="*80)
    
    # Test query
    test_requirements = {
        "num_guests": 7,
        "budget": 500,
        "timing": "dinner",
        "cuisine": "italian",
        "vibe": "romantic",
        "occasion": "anniversary"
    }
    
    print("\n📋 User Requirements:")
    for key, value in test_requirements.items():
        print(f"   {key}: {value}")
    
    # Search
    results = search_restaurants(test_requirements)
    
    # Display results
    if results:
        print("\n\n" + "="*80)
        print("🎯 TOP RECOMMENDATIONS")
        print("="*80)
        
        for i, rec in enumerate(results, 1):
            print(format_recommendation(rec, i))
    else:
        print("\n❌ No restaurants found matching your criteria")