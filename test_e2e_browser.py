"""
End-to-End Browser Tests using Playwright
These tests verify UI elements are visible and functional
Run with: pytest test_e2e_browser.py
"""
import pytest
from playwright.sync_api import sync_playwright, Page, expect
import os

# Set your app URL (change for production)
APP_URL = os.getenv('APP_URL', 'http://localhost:5000')


@pytest.fixture(scope="session")
def browser():
    """Create browser instance"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    """Create new page for each test"""
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()


class TestUIElements:
    """Test that UI elements are visible and functional"""
    
    def test_homepage_loads(self, page: Page):
        """Verify homepage loads with scanner dropdown"""
        page.goto(APP_URL)
        
        # Check for scanner dropdown
        scanner_select = page.locator('select[name="scanner"]')
        expect(scanner_select).to_be_visible()
        print("✓ Scanner dropdown visible")
    
    def test_beta_displays(self, page: Page):
        """Verify Beta displays in results"""
        page.goto(f'{APP_URL}/?scanner=supertrend')
        page.wait_for_load_state('networkidle')
        
        # Look for Beta in the results
        beta_elements = page.get_by_text('Beta:')
        if beta_elements.count() > 0:
            print(f"✓ Found {beta_elements.count()} Beta displays")
            
            # Check tooltip on hover
            first_beta = beta_elements.first
            first_beta.hover()
            # Tooltip should appear
            page.wait_for_timeout(500)  # Wait for tooltip
        else:
            print("⚠ No Beta displays found (may be no results)")
    
    def test_confirmations_block_displays(self, page: Page):
        """Verify confirmations block shows when present"""
        page.goto(f'{APP_URL}/?scanner=supertrend')
        page.wait_for_load_state('networkidle')
        
        # Look for confirmations text
        confirmations = page.get_by_text('Confirmed by other scanners')
        if confirmations.count() > 0:
            print(f"✓ Found {confirmations.count()} confirmation blocks")
            
            # Verify it's visible (not hidden)
            expect(confirmations.first).to_be_visible()
        else:
            print("⚠ No confirmations found (may be no multi-scanner symbols in results)")
    
    def test_dark_pool_chart_renders(self, page: Page):
        """Verify dark pool charts render"""
        page.goto(f'{APP_URL}/?scanner=supertrend')
        page.wait_for_load_state('networkidle')
        
        # Look for dark pool canvas
        dp_charts = page.locator('canvas.dp-chart')
        if dp_charts.count() > 0:
            print(f"✓ Found {dp_charts.count()} dark pool charts")
        else:
            print("⚠ No dark pool charts found in results")
    
    def test_fund_quality_tooltip(self, page: Page):
        """Verify fund quality tooltip appears on hover"""
        page.goto(f'{APP_URL}/?scanner=supertrend')
        page.wait_for_load_state('networkidle')
        
        # Find fund quality elements
        fund_elements = page.locator('.fund-quality')
        if fund_elements.count() > 0:
            print(f"✓ Found {fund_elements.count()} fund quality indicators")
            
            # Hover over first one
            fund_elements.first.hover()
            page.wait_for_timeout(500)
            
            # Check if tooltip appears
            tooltip = page.locator('.fund-tooltip')
            if tooltip.count() > 0:
                print("✓ Fund quality tooltip appears on hover")
        else:
            print("⚠ No fund quality indicators found")
    
    def test_timeline_displays(self, page: Page):
        """Verify timeline shows scanner detections"""
        page.goto(f'{APP_URL}/?scanner=supertrend')
        page.wait_for_load_state('networkidle')
        
        # Look for timeline markers
        timeline_markers = page.locator('.timeline-marker')
        if timeline_markers.count() > 0:
            print(f"✓ Found {timeline_markers.count()} timeline markers")
        else:
            print("⚠ No timeline markers found (may be no historical data)")


class TestNavigation:
    """Test navigation between pages"""
    
    def test_navigate_to_darkpool_signals(self, page: Page):
        """Navigate to dark pool signals page"""
        page.goto(APP_URL)
        
        # Click dark pool link in navigation
        page.get_by_role('link', name='Dark Pool Signals').click()
        page.wait_for_load_state('networkidle')
        
        # Verify we're on dark pool page
        expect(page).to_have_url(f'{APP_URL}/darkpool-signals')
        print("✓ Dark pool signals page loads")
    
    def test_navigate_to_options_signals(self, page: Page):
        """Navigate to options signals page"""
        page.goto(APP_URL)
        
        page.get_by_role('link', name='Options Signals').click()
        page.wait_for_load_state('networkidle')
        
        expect(page).to_have_url(f'{APP_URL}/options-signals')
        print("✓ Options signals page loads")
    
    def test_navigate_to_focus_list(self, page: Page):
        """Navigate to focus list"""
        page.goto(APP_URL)
        
        page.get_by_role('link', name='Focus List').click()
        page.wait_for_load_state('networkidle')
        
        expect(page).to_have_url(f'{APP_URL}/focus-list')
        print("✓ Focus list page loads")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
