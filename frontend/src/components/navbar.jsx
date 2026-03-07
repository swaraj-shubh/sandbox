import React, { useEffect, useState } from "react"
import { Link, useLocation, useNavigate } from "react-router-dom"
import { FaBars, FaTimes } from "react-icons/fa"
import { Button } from "@/components/ui/button"
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
} from "@/components/ui/navigation-menu"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu"

function Navbar() {
  const location = useLocation()
  const navigate = useNavigate()
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [role, setRole] = useState(null)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  useEffect(() => {
    const token = localStorage.getItem("token")
    const userRole = localStorage.getItem("role")
    setIsAuthenticated(!!token)
    setRole(userRole)
  }, [])

  const handleLogout = () => {
    localStorage.removeItem("token")
    localStorage.removeItem("role")
    setIsAuthenticated(false)
    setRole(null)
    navigate("/")
  }

  const getLinkClass = (path) => {
    return `px-4 py-2 rounded-md transition-colors hover:bg-[#c0392b] text-gray-100 hover:shadow-lg shadow-black hover:text-white hover:font-medium transition-transform duration-300 hover:scale-105 ${
      location.pathname === path ? "text-white font-bold" : ""
    }`
  }

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen)
  }

  return (
    <header className="bg-green-900 rounded-b-sm text-white shadow-2xl w-full sticky top-0 z-50">
      <div className="container flex items-center justify-between p-4 mx-auto">
        {/* Mobile menu button */}
        <div className="md:hidden flex items-center">
          <button
            onClick={toggleMobileMenu}
            className="text-white focus:outline-none"
            aria-label="Toggle menu"
          >
            {isMobileMenuOpen ? <FaTimes size={24} /> : <FaBars size={24} />}
          </button>
        </div>

        {/* Logo */}
        <Link
          to="/"
          className="hidden md:block text-lg font-semibold tracking-wide hover:opacity-90"
        >
          VAJRA
        </Link>

        {/* Desktop Navigation */}
        <NavigationMenu className="hidden md:block w-full">
          <NavigationMenuList className="flex gap-4 w-full items-center">
            <NavigationMenuItem>
              <NavigationMenuLink asChild>
                <Link to="/" className={getLinkClass("/")}>
                  Home
                </Link>
              </NavigationMenuLink>
            </NavigationMenuItem>

            <NavigationMenuItem>
              <NavigationMenuLink asChild>
                <Link to="/map" className={getLinkClass("/map")}>
                  Map
                </Link>
              </NavigationMenuLink>
            </NavigationMenuItem>

            {(role === "donor" || role === "admin") && (
              <NavigationMenuItem>
                <NavigationMenuLink asChild>
                  <Link
                    to="/restaurantDashboard"
                    className={getLinkClass("/restaurantDashboard")}
                  >
                    Donate
                  </Link>
                </NavigationMenuLink>
              </NavigationMenuItem>
            )}

            <div className="ml-auto flex items-center gap-4">
              <NavigationMenuItem>
                <NavigationMenuLink asChild>
                  <Link to="/profile" className={getLinkClass("/profile")}>
                    Profile
                  </Link>
                </NavigationMenuLink>
              </NavigationMenuItem>

              {role === "admin" && (
                <NavigationMenuItem>
                  <NavigationMenuLink asChild>
                    <Link to="/admin" className={getLinkClass("/admin")}>
                      Admin
                    </Link>
                  </NavigationMenuLink>
                </NavigationMenuItem>
              )}

              <NavigationMenuItem>
                {isAuthenticated ? (
                  <Button
                    variant="secondary"
                    onClick={handleLogout}
                    className="bg-white text-black font-semibold hover:bg-gray-200 transition-all duration-300"
                  >
                    Logout
                  </Button>
                ) : (
                  <NavigationMenuLink asChild>
                    <Link to="/auth" className={getLinkClass("/auth")}>
                      Login
                    </Link>
                  </NavigationMenuLink>
                )}
              </NavigationMenuItem>
            </div>
          </NavigationMenuList>
        </NavigationMenu>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <div className="md:hidden absolute top-16 left-0 right-0 bg-[#E74C3C] z-50 shadow-lg">
            <div className="flex flex-col p-4 space-y-4">
              <Link
                to="/"
                className={getLinkClass("/")}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Home
              </Link>

              <Link
                to="/map"
                className={getLinkClass("/map")}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Map
              </Link>

              {(role === "donor" || role === "admin") && (
                <Link
                  to="/restaurantDashboard"
                  className={getLinkClass("/restaurantDashboard")}
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  Donate
                </Link>
              )}

              <Link
                to="/profile"
                className={getLinkClass("/profile")}
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Profile
              </Link>

              {role === "admin" && (
                <Link
                  to="/admin"
                  className={getLinkClass("/admin")}
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  Admin
                </Link>
              )}

              {isAuthenticated ? (
                <Button
                  onClick={() => {
                    handleLogout()
                    setIsMobileMenuOpen(false)
                  }}
                  className="bg-white text-black font-semibold hover:bg-gray-200 transition-all duration-300 text-left"
                >
                  Logout
                </Button>
              ) : (
                <Link
                  to="/auth"
                  className={getLinkClass("/auth")}
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  Login
                </Link>
              )}
            </div>
          </div>
        )}
      </div>
    </header>
  )
}

export default Navbar
